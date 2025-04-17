#include "advection_diffusion_problem.hpp"
//
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <meltpooldg/level_set/advection_DG_operation.hpp>
#include <meltpooldg/level_set/advection_diffusion_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include "advection_diffusion_case.hpp"

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  void
  AdvectionDiffusionProblem<dim, number>::run()
  {
    initialize();

    while (!time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));
        /*
         * compute the advection velocity for the current time
         */
        compute_advection_velocity(
          *simulation_case->get_field_function("prescribed_velocity", "advection_diffusion"));

        advec_diff_operation->solve();
        /*
         *  do output if requested
         */
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

        if (simulation_case->parameters.amr.do_amr)
          refine_mesh();

        if (profiling_monitor && profiling_monitor->now())
          {
            profiling_monitor->print(scratch_data->get_pcout(1),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }
      }
    //... always print timing statistics
    if (profiling_monitor)
      {
        profiling_monitor->print(scratch_data->get_pcout(1),
                                 scratch_data->get_timer(),
                                 scratch_data->get_mpi_comm());
      }
    Journal::print_end(scratch_data->get_pcout(1));
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionProblem<dim, number>::setup_dof_system()
  {
    /*
     *  setup DoFHandler
     */
    FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.advec_diff.fe,
                                                dof_handler);
    FiniteElementUtils::distribute_dofs<dim, dim>(simulation_case->parameters.advec_diff.fe,
                                                  dof_handler_velocity);

    /*
     *  create the partititioning
     */
    scratch_data->create_partitioning();
    /*
     *  make hanging nodes and dirichlet constraints (Note: at the moment no time-dependent
     *  dirichlet constraints are supported)
     */
    simulation_case->set_time_boundary_conditions(
      simulation_case->parameters.time_stepping.start_time);

    if (simulation_case->parameters.advec_diff.fe.type != FiniteElementType::FE_DGQ)
      { // In a DG simulation no hanging node constraints are present and the boundray condtions are
        // enforced in weak form by changing the fluxes at the boundary
        MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim,
                                                                                           number>(
          *scratch_data,
          simulation_case->get_boundary_condition("dirichlet", "advection_diffusion"),
          simulation_case->get_periodic_bc(),
          advec_diff_dof_idx,
          advec_diff_hanging_nodes_dof_idx);
        if (simulation_case->parameters.advec_diff.implementation == "adaflo")
          MeltPoolDG::Constraints::make_DBC_and_HNC_and_merge_HNC_into_DBC<dim, number>(
            *scratch_data,
            simulation_case->get_boundary_condition("dirichlet", "advection_diffusion"),
            advec_diff_adaflo_dof_idx,
            advec_diff_hanging_nodes_dof_idx,
            false /*set inhomogeneities to zero*/);
        MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                        simulation_case->get_periodic_bc(),
                                                        velocity_dof_idx);
      }
    /*
     *  create the matrix-free object
     */
    if (simulation_case->parameters.advec_diff.fe.type != FiniteElementType::FE_DGQ)
      {
        scratch_data->build(false, false);
      }
    else
      {
        scratch_data->build(true, true);
      }

    if (advec_diff_operation) // TODO: better place
      advec_diff_operation->reinit();

    /*
     * print mesh information
     */
    CellMonitor<number>::add_info("advecDiff::cells",
                                  scratch_data->get_triangulation().n_global_active_cells(),
                                  scratch_data->get_min_cell_size(),
                                  scratch_data->get_max_cell_size());
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionProblem<dim, number>::initialize()
  {
    /*
     *  setup DoFHandler
     */
    dof_handler.reinit(*simulation_case->triangulation);
    dof_handler_velocity.reinit(*simulation_case->triangulation);

    /*
     *  setup scratch data
     */
    {
      scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
        simulation_case->mpi_communicator,
        simulation_case->parameters.base.verbosity_level,
        simulation_case->parameters.advec_diff.linear_solver.do_matrix_free);
      /*
       *  setup mapping
       */
      scratch_data->set_mapping(
        FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.advec_diff.fe));
      /*
       *  create quadrature rule
       */
      advec_diff_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.advec_diff.fe));

      advec_diff_dof_idx               = scratch_data->attach_dof_handler(dof_handler);
      advec_diff_hanging_nodes_dof_idx = scratch_data->attach_dof_handler(dof_handler);
      advec_diff_adaflo_dof_idx        = scratch_data->attach_dof_handler(dof_handler);
      velocity_dof_idx                 = scratch_data->attach_dof_handler(dof_handler_velocity);

      scratch_data->attach_constraint_matrix(constraints);
      scratch_data->attach_constraint_matrix(hanging_node_constraints);
      scratch_data->attach_constraint_matrix(hanging_node_constraints_with_zero_dirichlet);
      scratch_data->attach_constraint_matrix(hanging_node_constraints_velocity);
    }

    setup_dof_system();

    /*
     *  initialize the time iterator
     */
    time_iterator =
      std::make_unique<TimeIterator<number>>(simulation_case->parameters.time_stepping);

    if (simulation_case->parameters.advec_diff.implementation == "meltpooldg")
      {
        if (simulation_case->parameters.advec_diff.fe.type != FiniteElementType::FE_DGQ)
          {
            advec_diff_operation = std::make_unique<AdvectionDiffusionOperation<dim, number>>(
              *scratch_data,
              simulation_case->get_boundary_condition("dirichlet", "advection_diffusion"),
              simulation_case->parameters.advec_diff,
              *time_iterator,
              advection_velocity,
              advec_diff_dof_idx,
              advec_diff_hanging_nodes_dof_idx,
              advec_diff_quad_idx,
              velocity_dof_idx);

            dynamic_cast<AdvectionDiffusionOperation<dim, number> *>(advec_diff_operation.get())
              ->set_inflow_outflow_bc(
                simulation_case->get_boundary_condition("inflow_outflow", "advection_diffusion"));
          }
        else
          {
            advec_diff_operation = std::make_unique<AdvectionDGOperation<dim, number>>(
              *scratch_data,
              simulation_case->parameters.advec_diff,
              *time_iterator,
              advection_velocity,
              advec_diff_dof_idx,
              advec_diff_quad_idx,
              velocity_dof_idx,
              simulation_case->get_boundary_condition_manager("advection_diffusion"),
              simulation_case->get_field_function("prescribed_velocity", "advection_diffusion"),
              true);

            /*In the DG case the boundary conditions are applied weakly inside the operator*/
          }
        advec_diff_operation->reinit();
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if (simulation_case->parameters.advec_diff.implementation == "adaflo")
      {
        AssertThrow(
          simulation_case->get_boundary_condition("inflow_outflow", "advection_diffusion").empty(),
          ExcMessage(
            "Inflow/outflow boundary condition not supported from the adaflo implementation."));

        advec_diff_operation = std::make_unique<AdvectionDiffusionOperationAdaflo<dim, number>>(
          *scratch_data,
          *time_iterator,
          advection_velocity,
          advec_diff_adaflo_dof_idx,
          advec_diff_dof_idx,
          advec_diff_quad_idx,
          velocity_dof_idx,
          simulation_case->parameters.time_stepping,
          simulation_case->parameters.advec_diff,
          *simulation_case->get_boundary_condition_manager("advection_diffusion"));
        advec_diff_operation->reinit();
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     *  set initial conditions for the advected field
     */
    compute_advection_velocity(
      *simulation_case->get_field_function("prescribed_velocity", "advection_diffusion"));
    advec_diff_operation->set_initial_condition(
      *simulation_case->get_initial_condition("advection_diffusion"));
    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(advec_diff_dof_idx),
                                                   simulation_case->parameters.output,
                                                   simulation_case->parameters.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(
                                                     advec_diff_dof_idx),
                                                   scratch_data->get_pcout(2));
    /*
     *  initialize profiling
     */
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionProblem<dim, number>::compute_advection_velocity(
    Function<dim, number> &advec_func)
  {
    scratch_data->initialize_dof_vector(advection_velocity, velocity_dof_idx);
    /*
     *  set the current time to the advection field function
     */
    advec_func.set_time(time_iterator->get_current_time());
    /*
     *  interpolate the values of the advection velocity
     */
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(velocity_dof_idx),
                                     advec_func,
                                     advection_velocity);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionProblem<dim, number>::output_results(const unsigned int time_step,
                                                         const number       current_time)
  {
    if (!post_processor->is_output_timestep(time_step, current_time) &&
        !simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &data_out) {
      advec_diff_operation->attach_output_vectors(data_out);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);
      data_out.add_data_vector(scratch_data->get_dof_handler(velocity_dof_idx),
                               advection_velocity,
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);
    };

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(),
      current_time,
      simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, current_time);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionProblem<dim, number>::refine_mesh()
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(simulation_case->triangulation->n_active_cells());

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
        simulation_case->parameters.amr.upper_perc_to_refine,
        simulation_case->parameters.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      advec_diff_operation->attach_vectors(vectors);
    };

    const auto post = [&]() { constraints.distribute(advec_diff_operation->get_advected_field()); };

    const auto setup_dof_system = [&]() { this->setup_dof_system(); };

    AMR::refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                      attach_vectors,
                                      post,
                                      setup_dof_system,
                                      simulation_case->parameters.amr,
                                      dof_handler,
                                      time_iterator->get_current_time_step_number());
  }

  template class AdvectionDiffusionProblem<1, double>;
  template class AdvectionDiffusionProblem<2, double>;
  template class AdvectionDiffusionProblem<3, double>;
} // namespace MeltPoolDG::LevelSet

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::LevelSet::AdvectionDiffusionCaseParameters<double>,
                           MeltPoolDG::LevelSet::AdvectionDiffusionCase,
                           MeltPoolDG::LevelSet::AdvectionDiffusionProblem>(argc, argv, mpi_comm);
  return 0;
}
