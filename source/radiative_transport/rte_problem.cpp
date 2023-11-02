#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/numerics/error_estimator.h>

#include <meltpooldg/radiative_transport/rte_problem.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <memory>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim>
  void
  RadiativeTransportProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    while (!time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout());

        compute_heaviside(*base_in->get_initial_condition("prescribed_heaviside"));

        rte_operation->solve();

        // calculate the source field and output the results to vtk files.
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time(),
                       base_in);
      }

    //... print timing statistics
    if (base_in->parameters.profiling.enable)
      {
        scratch_data->get_timer().print_wall_time_statistics(scratch_data->get_mpi_comm());
        scratch_data->get_pcout() << std::endl;
        IterationMonitor::print(scratch_data->get_pcout());
        scratch_data->get_pcout() << std::endl;
        DoFMonitor::print(scratch_data->get_pcout());
      }
    Journal::print_end(scratch_data->get_pcout());
  }

  template <int dim>
  std::string
  RadiativeTransportProblem<dim>::get_name()
  {
    return "radiative_transport";
  }

  template <int dim>
  void
  RadiativeTransportProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  setup scratch data
     */
    scratch_data =
      std::make_shared<ScratchData<dim>>(base_in->mpi_communicator,
                                         base_in->parameters.base.verbosity_level,
                                         base_in->parameters.rte.linear_solver.do_matrix_free);
    /*
     *  setup mapping
     */
    if (base_in->parameters.base.do_simplex)
      scratch_data->set_mapping(MappingFE<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree)));
    else
      scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));

    /*
     *  setup DoFHandler
     */
    dof_handler.reinit(*base_in->triangulation);
    dof_handler_heaviside.reinit(*base_in->triangulation);

    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler_heaviside);

    /*
     * attach constraints
     */
    rte_dof_idx               = scratch_data->attach_constraint_matrix(constraints_dirichlet);
    rte_hanging_nodes_dof_idx = scratch_data->attach_constraint_matrix(hanging_node_constraints);
    hs_dof_idx = scratch_data->attach_constraint_matrix(hanging_node_constraints_heaviside);

    /*
     *  create quadrature rule
     */
    if (base_in->parameters.base.do_simplex)
      rte_quad_idx =
        scratch_data->attach_quadrature(QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
    else
      rte_quad_idx =
        scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));

    setup_dof_system(base_in, false);

    /*
     *  initialize the time stepping scheme
     */
    time_iterator = std::make_shared<TimeIterator<double>>(base_in->parameters.time_stepping);

    /*
     *  set initial conditions of the heaviside field
     */
    compute_heaviside(*base_in->get_initial_condition("prescribed_heaviside"));

    /*
     * initialize the RTE operation
     */
    rte_operation = std::make_shared<RadiativeTransportOperation<dim>>(*scratch_data,
                                                                       base_in->parameters.rte,
                                                                       heaviside,
                                                                       rte_dof_idx,
                                                                       rte_hanging_nodes_dof_idx,
                                                                       rte_quad_idx,
                                                                       hs_dof_idx);

    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(rte_dof_idx),
                                           base_in->parameters.paraview,
                                           base_in->parameters.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(rte_dof_idx),
                                           scratch_data->get_pcout(1));

    /*
     *    Do initial refinement steps if requested
     */
    if (base_in->parameters.amr.do_amr && base_in->parameters.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
        {
          std::ostringstream str;
          str << " #dofs intensity: " << rte_operation->get_intensity().size();
          Journal::print_line(scratch_data->get_pcout(), str.str(), "RTE");
          refine_mesh(base_in);
        }

    /*
     *  output results of initialization
     */
    output_results(0, base_in->parameters.time_stepping.start_time, base_in);
  }

  template <int dim>
  void
  RadiativeTransportProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                                   const bool                           do_reinit)
  {
    if (base_in->parameters.base.do_simplex)
      {
        dof_handler.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
        dof_handler_heaviside.distribute_dofs(
          FESystem<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree)));
      }
    else
      {
        dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
        dof_handler_heaviside.distribute_dofs(
          FESystem<dim>(FE_Q<dim>(base_in->parameters.base.degree)));
      }

    /*
     *  create partitioning
     */
    scratch_data->create_partitioning();

    /*
     *  create AffineConstraints
     */
    base_in->attach_boundary_condition("intensity");
    MeltPoolDG::UtilityFunctions::setup_constraints<dim>(*scratch_data,
                                                         base_in->get_dirichlet_bc("intensity"),
                                                         base_in->get_periodic_bc(),
                                                         rte_dof_idx,
                                                         rte_hanging_nodes_dof_idx);

    /*
     *  create the matrix-free object
     */
    scratch_data->build();

    if (do_reinit)
      rte_operation->reinit();
  }

  template <int dim>
  void
  RadiativeTransportProblem<dim>::compute_heaviside(Function<dim> &heaviside_func)
  {
    scratch_data->initialize_dof_vector(heaviside, hs_dof_idx);
    heaviside_func.set_time(time_iterator->get_current_time());
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     dof_handler_heaviside,
                                     heaviside_func,
                                     heaviside);
  }

  /*
   *  This function is to create paraview output
   */
  template <int dim>
  void
  RadiativeTransportProblem<dim>::output_results(const unsigned int                   time_step,
                                                 const double                         time,
                                                 std::shared_ptr<SimulationBase<dim>> base_in)
  {
    if (!post_processor->now(time_step, time) &&
        !base_in->parameters.paraview.do_user_defined_postprocessing)
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      scratch_data->initialize_dof_vector(heat_source, rte_dof_idx);
      rte_operation->compute_heat_source(heat_source, rte_dof_idx, true);
      rte_operation->attach_output_vectors(data_out);
      data_out.add_data_vector(dof_handler_heaviside, heaviside, "prescribed_heaviside");
      data_out.add_data_vector(dof_handler, heat_source, "heat_source");
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         time,
                                         base_in->parameters.paraview.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (base_in->parameters.paraview.do_user_defined_postprocessing)
      base_in->do_postprocessing(generic_data_out);

    // paraview postprocessing
    post_processor->process(time_step, generic_data_out, time);
  }

  /*
   *  perform mesh refinement
   */
  template <int dim>
  void
  RadiativeTransportProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(rte_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(rte_operation->get_intensity());
      constraints_dirichlet.distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      KellyErrorEstimator<dim>::estimate(scratch_data->get_dof_handler(rte_dof_idx),
                                         scratch_data->get_face_quadrature(rte_dof_idx),
                                         {},
                                         locally_relevant_solution,
                                         estimated_error_per_cell);

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        base_in->parameters.amr.upper_perc_to_refine,
        base_in->parameters.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      rte_operation->attach_vectors(vectors);
    };

    const auto post = [&]() { rte_operation->distribute_constraints(); };

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in, true); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 base_in->parameters.amr,
                                 dof_handler,
                                 time_iterator->get_current_time_step_number());
  }

  template class RadiativeTransportProblem<1>;
  template class RadiativeTransportProblem<2>;
  template class RadiativeTransportProblem<3>;
} // namespace MeltPoolDG::RadiativeTransport
