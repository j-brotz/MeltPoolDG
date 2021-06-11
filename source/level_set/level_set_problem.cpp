#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/interface/problem_base.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_problem.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/postprocessor.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  void
  LevelSetProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    while (!time_iterator.is_finished())
      {
        const double dt = time_iterator.get_next_time_increment();
        scratch_data->get_pcout() << "| ls: t= " << std::setw(10) << std::left
                                  << time_iterator.get_current_time();
        compute_advection_velocity(*base_in->get_advection_field("level_set"));

        if (evaporation_operation)
          {
            /**
             * If evaporative mass flux is considered the interface velocity will be modified.
             * Note that the normal vector is used from the old step.
             */
            level_set_operation.update_normal_vector();
            evaporation_operation->compute_evaporation_velocity();
            /**
             * compute advection velocity of the interface
             */
            advection_velocity += evaporation_operation->get_velocity();
          }
        level_set_operation.solve(dt, advection_velocity);


        // do paraview output if requested
        output_results(time_iterator.get_current_time_step_number(),
                       time_iterator.get_current_time(),
                       base_in);

        if (base_in->parameters.amr.do_amr)
          refine_mesh(base_in);
      }
  }

  template <int dim>
  std::string
  LevelSetProblem<dim>::get_name()
  {
    return "level_set_problem";
  }

  /*
   *  This function initials the relevant scratch data
   *  for the computation of the level set problem
   */
  template <int dim>
  void
  LevelSetProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  setup scratch data
     */
    scratch_data = std::make_shared<ScratchData<dim>>(base_in->mpi_communicator,
                                                      base_in->parameters.base.verbosity_level,
                                                      /* do_matrix_free */ true);
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
    dof_handler_velocity.reinit(*base_in->triangulation);

    this->ls_hanging_nodes_dof_idx = scratch_data->attach_dof_handler(dof_handler);
    this->ls_dof_idx               = scratch_data->attach_dof_handler(dof_handler);
    ls_zero_bc_idx                 = scratch_data->attach_dof_handler(dof_handler);
    vel_dof_idx                    = scratch_data->attach_dof_handler(dof_handler_velocity);

    scratch_data->attach_constraint_matrix(hanging_node_constraints);
    scratch_data->attach_constraint_matrix(constraints_dirichlet);
    scratch_data->attach_constraint_matrix(hanging_node_constraints_with_zero_dirichlet);
    scratch_data->attach_constraint_matrix(hanging_node_constraints_velocity);

    /*
     *  create quadrature rule
     */

    if (base_in->parameters.base.do_simplex)
      ls_quad_idx =
        scratch_data->attach_quadrature(QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
    else
      ls_quad_idx =
        scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));

    // TODO: only do once!
    if (base_in->parameters.base.do_simplex)
      scratch_data->attach_quadrature(QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
    else
      scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));
    /*
     *  initialize the time iterator
     */
    time_iterator.initialize(
      TimeIteratorData<double>{base_in->parameters.time_stepping.start_time,
                               base_in->parameters.time_stepping.end_time,
                               base_in->parameters.time_stepping.time_step_size,
                               100000,
                               false});

    setup_dof_system(base_in, false);

    /*
     * initialize the levelset operation class
     */

    level_set_operation.initialize(scratch_data,
                                   base_in,
                                   ls_dof_idx,
                                   ls_hanging_nodes_dof_idx,
                                   ls_quad_idx,
                                   reinit_dof_idx,
                                   reinit_dof_idx,
                                   curv_dof_idx,
                                   normal_dof_idx,
                                   vel_dof_idx,
                                   ls_zero_bc_idx);

    /*
     * set initial conditions
     */
    compute_advection_velocity(*base_in->get_advection_field("level_set"));

    /*
     * configure level set with evaporation if requested
     */
    if (base_in->parameters.base.problem_name == "level_set_with_evaporation")
      {
        evaporation_operation = std::make_shared<Evaporation::EvaporationOperation<dim>>(
          scratch_data,
          level_set_operation.get_level_set_as_heaviside(),
          level_set_operation.get_normal_vector(),
          base_in,
          normal_dof_idx,
          vel_dof_idx,
          ls_hanging_nodes_dof_idx,
          ls_quad_idx);

        /**
         *  set evaporative mass flux constant
         */
        evaporation_operation->get_evaporative_mass_flux() =
          base_in->parameters.evapor.evaporative_mass_flux;

        /**
         * register evaporative mass flux model
         */
        evaporation_operation->register_evaporative_mass_flux_model(
          base_in->parameters.recoil,
          level_set_operation.get_distance_to_level_set(),
          base_in->parameters.reinit.constant_epsilon,
          base_in->parameters.reinit.scale_factor_epsilon);
      }
    /**
     * set the initial velocity field
     */
    level_set_operation.set_initial_condition(*base_in->get_initial_condition("level_set"),
                                              advection_velocity);
    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(ls_dof_idx),
                                           base_in->parameters.paraview,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(ls_dof_idx));

    output_results(0, base_in->parameters.time_stepping.start_time, base_in);
    /*
     *    Do initial refinement steps if requested
     */
    if (base_in->parameters.amr.do_amr && base_in->parameters.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
        {
          scratch_data->get_pcout()
            << "cycle: " << i << " n_dofs: " << dof_handler.n_dofs() << "(ls)" << std::endl;
          refine_mesh(base_in);
          /*
           *  set initial conditions after initial AMR
           */
          compute_advection_velocity(*base_in->get_advection_field("level_set"));
          level_set_operation.set_initial_condition(*base_in->get_initial_condition("level_set"),
                                                    advection_velocity);
        }
  }

  template <int dim>
  void
  LevelSetProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                         const bool                           do_reinit)
  {
    if (base_in->parameters.base.do_simplex)
      {
        dof_handler.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
        dof_handler_velocity.distribute_dofs(
          FESystem<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree), dim));
      }
    else
      {
        dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
        dof_handler_velocity.distribute_dofs(
          FESystem<dim>(FE_Q<dim>(base_in->parameters.base.degree), dim));
      }
    /*
     *  create partitioning
     */
    scratch_data->create_partitioning();
    /*
     *  make hanging nodes constraints
     */

    /*
     *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
     *  dirichlet constraints are supported)
     */
    hanging_node_constraints.clear();
    hanging_node_constraints.reinit(
      scratch_data->get_locally_relevant_dofs(ls_hanging_nodes_dof_idx));
    DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
    hanging_node_constraints.close();

    hanging_node_constraints_velocity.clear();
    hanging_node_constraints_velocity.reinit(scratch_data->get_locally_relevant_dofs(vel_dof_idx));
    DoFTools::make_hanging_node_constraints(dof_handler_velocity,
                                            hanging_node_constraints_velocity);
    hanging_node_constraints_velocity.close();

    constraints_dirichlet.clear();
    constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs(ls_dof_idx));
    for (const auto &bc :
         base_in->get_dirichlet_bc("level_set")) // @todo: add name of bc at a more central place
      {
        dealii::VectorTools::interpolate_boundary_values(
          scratch_data->get_mapping(), dof_handler, bc.first, *bc.second, constraints_dirichlet);
      }
    constraints_dirichlet.close();
    constraints_dirichlet.merge(
      hanging_node_constraints,
      AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

    hanging_node_constraints_with_zero_dirichlet.clear();
    hanging_node_constraints_with_zero_dirichlet.reinit(
      scratch_data->get_locally_relevant_dofs(ls_zero_bc_idx));
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints_with_zero_dirichlet);
    for (const auto &bc :
         base_in->get_dirichlet_bc("level_set")) // @todo: add name of bc at a more central place
      {
        dealii::DoFTools::make_zero_boundary_constraints(
          dof_handler, bc.first, hanging_node_constraints_with_zero_dirichlet);
      }
    hanging_node_constraints_with_zero_dirichlet.close();
    /*
     *  create the matrix-free object
     */
    scratch_data->build();

    if (do_reinit)
      {
        level_set_operation.reinit();
        if (evaporation_operation)
          evaporation_operation->reinit();
      }
  }

  template <int dim>
  void
  LevelSetProblem<dim>::compute_advection_velocity(Function<dim> &advec_func)
  {
    scratch_data->initialize_dof_vector(advection_velocity, vel_dof_idx);
    /*
     *  set the current time to the advection field function
     */
    advec_func.set_time(time_iterator.get_current_time());

    dealii::VectorTools::project(scratch_data->get_mapping(),
                                 dof_handler_velocity,
                                 hanging_node_constraints_velocity,
                                 scratch_data->get_quadrature(),
                                 advec_func,
                                 advection_velocity);
  }
  /*
   *  This function is to create paraview output
   */
  template <int dim>
  void
  LevelSetProblem<dim>::output_results(const unsigned int                   time_step,
                                       const double                         time,
                                       std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      level_set_operation.attach_output_vectors(data_out);
      if (evaporation_operation)
        evaporation_operation->attach_output_vectors(data_out);
      /*
       *  output advection velocity
       */
      MeltPoolDG::VectorTools::update_ghost_values(advection_velocity);
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(dof_handler_velocity,
                               advection_velocity,
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(), time);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    base_in->do_postprocessing(generic_data_out);

    // paraview postprocessing
    post_processor->process(time_step, attach_output_vectors, time);
  }

  /*
   *  perform mesh refinement
   */
  template <int dim>
  void
  LevelSetProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement =
      [&](parallel::distributed::Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(level_set_operation.get_level_set());
      constraints_dirichlet.distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      for (unsigned int i = 0; i < locally_relevant_solution.locally_owned_size(); ++i)
        locally_relevant_solution.local_element(i) =
          (1.0 -
           locally_relevant_solution.local_element(i) * locally_relevant_solution.local_element(i));

      locally_relevant_solution.update_ghost_values();

      dealii::VectorTools::integrate_difference(scratch_data->get_dof_handler(ls_dof_idx),
                                                locally_relevant_solution,
                                                Functions::ZeroFunction<dim>(),
                                                estimated_error_per_cell,
                                                scratch_data->get_quadrature(ls_quad_idx),
                                                dealii::VectorTools::L2_norm);

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        base_in->parameters.amr.upper_perc_to_refine,
        base_in->parameters.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      level_set_operation.attach_vectors(vectors);
      if (evaporation_operation)
        evaporation_operation->attach_vectors(vectors);
    };

    const auto post = [&]() {
      constraints_dirichlet.distribute(level_set_operation.get_level_set());
      hanging_node_constraints.distribute(level_set_operation.get_level_set_as_heaviside());
    };

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 base_in->parameters.amr,
                                 dof_handler,
                                 time_iterator.get_current_time_step_number());
  }

  template class LevelSetProblem<1>;
  template class LevelSetProblem<2>;
  template class LevelSetProblem<3>;
} // namespace MeltPoolDG::LevelSet
