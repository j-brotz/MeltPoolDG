#include <meltpooldg/advection_diffusion/advection_diffusion_problem.hpp>
//

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/numerics/error_estimator.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_adaflo_wrapper.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::AdvectionDiffusion
{
  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    while (!time_iterator.is_finished())
      {
        const double dt = time_iterator.get_next_time_increment();
        time_iterator.print_me(scratch_data->get_pcout());
        /*
         * compute the advection velocity for the current time
         */
        compute_advection_velocity(*base_in->get_advection_field("advection_diffusion"));
        advec_diff_operation->solve(dt, advection_velocity);
        /*
         *  do paraview output if requested
         */
        output_results(time_iterator.get_current_time_step_number(),
                       time_iterator.get_current_time(),
                       base_in);

        if (base_in->parameters.amr.do_amr)
          {
            refine_mesh(base_in);
          }
      }
    Journal::print_end(scratch_data->get_pcout());
  }

  template <int dim>
  std::string
  AdvectionDiffusionProblem<dim>::get_name()
  {
    return "advection-diffusion problem";
  }

  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  setup DoFHandler
     */
    dof_handler.distribute_dofs(*fe);
    dof_handler_velocity.distribute_dofs(*fe_velocity);

    /*
     *  create the partititioning
     */
    scratch_data->create_partitioning();
    /*
     *  make hanging nodes and dirichlet constraints (Note: at the moment no time-dependent
     *  dirichlet constraints are supported)
     */
    base_in->register_operation("advection_diffusion"); //@todo move to a more central place
    MeltPoolDG::UtilityFunctions::setup_constraints<dim>(*scratch_data,
                                                         base_in->get_dirichlet_bc(
                                                           "advection_diffusion"),
                                                         base_in->get_periodic_bc(),
                                                         advec_diff_dof_idx,
                                                         advec_diff_hanging_nodes_dof_idx);
    MeltPoolDG::UtilityFunctions::setup_and_merge_constraints<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("advection_diffusion"),
      advec_diff_adaflo_dof_idx,
      advec_diff_hanging_nodes_dof_idx,
      false /*set inhomogeneities to zero*/);
    MeltPoolDG::UtilityFunctions::setup_constraints<dim>(*scratch_data,
                                                         base_in->get_periodic_bc(),
                                                         velocity_dof_idx);
    /*
     *  create the matrix-free object
     */
    scratch_data->build();

    if (advec_diff_operation) // TODO: better place
      advec_diff_operation->reinit();

    /*
     * print mesh information
     */
    Journal::print_mesh_information<dim>(*scratch_data, 1);
  }

  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  setup DoFHandler
     */
    dof_handler.reinit(*base_in->triangulation);
    dof_handler_velocity.reinit(*base_in->triangulation);

    if (base_in->parameters.base.do_simplex)
      {
        fe = std::make_unique<FE_SimplexP<dim>>(base_in->parameters.base.degree);
        fe_velocity =
          std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(base_in->parameters.base.degree), dim);
      }
    else
      {
        fe = std::make_unique<FE_Q<dim>>(base_in->parameters.base.degree);
        fe_velocity =
          std::make_unique<FESystem<dim>>(FE_Q<dim>(base_in->parameters.base.degree), dim);
      }

    /*
     *  setup scratch data
     */
    {
      scratch_data = std::make_shared<ScratchData<dim>>(
        base_in->mpi_communicator,
        base_in->parameters.base.verbosity_level,
        base_in->parameters.advec_diff.linear_solver.do_matrix_free);
      /*
       *  setup mapping
       */
      if (base_in->parameters.base.do_simplex)
        scratch_data->set_mapping(
          MappingFE<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree)));
      else
        scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));
      /*
       *  create quadrature rule
       */
      if (base_in->parameters.base.do_simplex)
        advec_diff_quad_idx = scratch_data->attach_quadrature(
          QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
      else
        advec_diff_quad_idx =
          scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));

      advec_diff_dof_idx               = scratch_data->attach_dof_handler(dof_handler);
      advec_diff_hanging_nodes_dof_idx = scratch_data->attach_dof_handler(dof_handler);
      advec_diff_adaflo_dof_idx        = scratch_data->attach_dof_handler(dof_handler);
      velocity_dof_idx                 = scratch_data->attach_dof_handler(dof_handler_velocity);

      scratch_data->attach_constraint_matrix(constraints);
      scratch_data->attach_constraint_matrix(hanging_node_constraints);
      scratch_data->attach_constraint_matrix(hanging_node_constraints_with_zero_dirichlet);
      scratch_data->attach_constraint_matrix(hanging_node_constraints_velocity);
    }

    setup_dof_system(base_in);

    /*
     *  initialize the time iterator
     */
    time_iterator.initialize(base_in->parameters.time_stepping);

    if (base_in->parameters.advec_diff.implementation == "meltpooldg")
      {
        advec_diff_operation = std::make_shared<AdvectionDiffusionOperation<dim>>();

        advec_diff_operation->initialize(scratch_data,
                                         base_in->parameters.advec_diff,
                                         advec_diff_dof_idx,
                                         advec_diff_hanging_nodes_dof_idx,
                                         advec_diff_quad_idx,
                                         velocity_dof_idx);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if (base_in->parameters.advec_diff.implementation == "adaflo")
      {
        AssertThrow(base_in->parameters.advec_diff.linear_solver.do_matrix_free,
                    ExcNotImplemented());
        advec_diff_operation =
          std::make_shared<AdvectionDiffusionOperationAdaflo<dim>>(*scratch_data,
                                                                   advec_diff_adaflo_dof_idx,
                                                                   advec_diff_dof_idx,
                                                                   advec_diff_quad_idx,
                                                                   velocity_dof_idx,
                                                                   base_in);
        advec_diff_operation->reinit();
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     *  set initial conditions for the advected field
     */
    compute_advection_velocity(*base_in->get_advection_field("advection_diffusion"));
    advec_diff_operation->set_initial_condition(
      *base_in->get_initial_condition("advection_diffusion"), advection_velocity);
    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(advec_diff_dof_idx),
                                           base_in->parameters.paraview,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(advec_diff_dof_idx),
                                           scratch_data->get_pcout(1));
  }

  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::compute_advection_velocity(Function<dim> &advec_func)
  {
    scratch_data->initialize_dof_vector(advection_velocity, velocity_dof_idx);
    /*
     *  set the current time to the advection field function
     */
    advec_func.set_time(time_iterator.get_current_time());
    /*
     *  interpolate the values of the advection velocity
     */
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(velocity_dof_idx),
                                     advec_func,
                                     advection_velocity);
  }

  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::output_results(const unsigned int                   time_step,
                                                 const double                         current_time,
                                                 std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      advec_diff_operation->attach_output_vectors(data_out);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);
      data_out.add_data_vector(scratch_data->get_dof_handler(velocity_dof_idx),
                               advection_velocity,
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping());
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    base_in->do_postprocessing(generic_data_out);

    // paraview postprocessing
    post_processor->process(time_step, attach_output_vectors, current_time);
  }

  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement =
      [&](parallel::distributed::Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(advec_diff_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(
        advec_diff_operation->get_advected_field());
      constraints.distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      KellyErrorEstimator<dim>::estimate(scratch_data->get_mapping(),
                                         scratch_data->get_dof_handler(advec_diff_dof_idx),
                                         scratch_data->get_face_quadrature(advec_diff_quad_idx),
                                         {}, // empty means estimate the error based on the
                                             // generalized Poisson equation with dirichlet bc
                                         locally_relevant_solution,
                                         estimated_error_per_cell);
      auto vec =
        Utilities::MPI::gather(scratch_data->get_mpi_comm(), estimated_error_per_cell.l2_norm());

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        base_in->parameters.amr.upper_perc_to_refine,
        base_in->parameters.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      advec_diff_operation->attach_vectors(vectors);
    };

    const auto post = [&]() { constraints.distribute(advec_diff_operation->get_advected_field()); };

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 base_in->parameters.amr,
                                 dof_handler,
                                 time_iterator.get_current_time_step_number());
  }

  template class AdvectionDiffusionProblem<1>;
  template class AdvectionDiffusionProblem<2>;
  template class AdvectionDiffusionProblem<3>;
} // namespace MeltPoolDG::AdvectionDiffusion
