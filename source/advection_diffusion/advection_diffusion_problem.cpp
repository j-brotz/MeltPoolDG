#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_adaflo_wrapper.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_problem.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    while (!time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout());
        /*
         * compute the advection velocity for the current time
         */
        compute_advection_velocity(*base_in->get_advection_field("advection_diffusion"));

        advec_diff_operation->solve();
        /*
         *  do output if requested
         */
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time(),
                       base_in);

        if (base_in->parameters.amr.do_amr)
          {
            refine_mesh(base_in);
          }

        if (profiling_monitor && profiling_monitor->now())
          {
            profiling_monitor->print(scratch_data->get_pcout(),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }
      }
    //... always print timing statistics
    if (profiling_monitor)
      {
        profiling_monitor->print(scratch_data->get_pcout(),
                                 scratch_data->get_timer(),
                                 scratch_data->get_mpi_comm());
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
    FiniteElementUtils::distribute_dofs<dim, 1>(base_in->parameters.base.fe, dof_handler);
    FiniteElementUtils::distribute_dofs<dim, dim>(base_in->parameters.base.fe,
                                                  dof_handler_velocity);

    /*
     *  create the partititioning
     */
    scratch_data->create_partitioning();
    /*
     *  make hanging nodes and dirichlet constraints (Note: at the moment no time-dependent
     *  dirichlet constraints are supported)
     */
    base_in->attach_boundary_condition("advection_diffusion"); //@todo move to a more central place
    MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_BC_into_DBC<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("advection_diffusion"),
      base_in->get_periodic_bc(),
      advec_diff_dof_idx,
      advec_diff_hanging_nodes_dof_idx);
    MeltPoolDG::Constraints::make_DBC_and_HNC_and_merge_HNC_into_DBC<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("advection_diffusion"),
      advec_diff_adaflo_dof_idx,
      advec_diff_hanging_nodes_dof_idx,
      false /*set inhomogeneities to zero*/);
    MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                    base_in->get_periodic_bc(),
                                                    velocity_dof_idx);
    /*
     *  create the matrix-free object
     */
    scratch_data->build(false, false);

    if (advec_diff_operation) // TODO: better place
      advec_diff_operation->reinit();

    /*
     * print mesh information
     */
    {
      ScopedName sc("advecDiff::cells");
      CellMonitor::add_info(sc,
                            scratch_data->get_triangulation().n_global_active_cells(),
                            scratch_data->get_min_cell_size(),
                            scratch_data->get_max_cell_size());
    }
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

    /*
     *  setup scratch data
     */
    {
      scratch_data = std::make_shared<ScratchData<dim>>(
        base_in->mpi_communicator,
        base_in->parameters.base.verbosity_level,
        base_in->parameters.ls.advec_diff.linear_solver.do_matrix_free);
      /*
       *  setup mapping
       */
      scratch_data->set_mapping(
        FiniteElementUtils::create_mapping<dim>(base_in->parameters.base.fe));
      /*
       *  create quadrature rule
       */
      advec_diff_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(base_in->parameters.base.fe));

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
    time_iterator = std::make_shared<TimeIterator<double>>(base_in->parameters.time_stepping);

    if (base_in->parameters.ls.advec_diff.implementation == "meltpooldg")
      {
        advec_diff_operation =
          std::make_shared<AdvectionDiffusionOperation<dim>>(*scratch_data,
                                                             base_in->parameters.ls.advec_diff,
                                                             *time_iterator,
                                                             advection_velocity,
                                                             advec_diff_dof_idx,
                                                             advec_diff_hanging_nodes_dof_idx,
                                                             advec_diff_quad_idx,
                                                             velocity_dof_idx);
        advec_diff_operation->reinit();

        dynamic_cast<AdvectionDiffusionOperation<dim> *>(advec_diff_operation.get())
          ->set_inflow_outflow_bc(base_in->get_bc("advection_diffusion")->inflow_outflow_bc);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if (base_in->parameters.ls.advec_diff.implementation == "adaflo")
      {
        AssertThrow(
          base_in->get_bc("advection_diffusion")->inflow_outflow_bc.empty(),
          ExcMessage(
            "Inflow/outflow boundary condition not supported from the adaflo implementation."));

        advec_diff_operation =
          std::make_shared<AdvectionDiffusionOperationAdaflo<dim>>(*scratch_data,
                                                                   *time_iterator,
                                                                   advection_velocity,
                                                                   advec_diff_adaflo_dof_idx,
                                                                   advec_diff_dof_idx,
                                                                   advec_diff_quad_idx,
                                                                   velocity_dof_idx,
                                                                   base_in);
        advec_diff_operation->reinit();

        // TODO: add assert for inflow/outflow BC that this is not supported
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     *  set initial conditions for the advected field
     */
    compute_advection_velocity(*base_in->get_advection_field("advection_diffusion"));
    advec_diff_operation->set_initial_condition(
      *base_in->get_initial_condition("advection_diffusion"));
    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(advec_diff_dof_idx),
                                           base_in->parameters.output,
                                           base_in->parameters.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(advec_diff_dof_idx),
                                           scratch_data->get_pcout(1));
    /*
     *  initialize profiling
     */
    if (base_in->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<double>>(base_in->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::compute_advection_velocity(Function<dim> &advec_func)
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

  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::output_results(const unsigned int                   time_step,
                                                 const double                         current_time,
                                                 std::shared_ptr<SimulationBase<dim>> base_in)
  {
    if (!post_processor->is_output_timestep(time_step, current_time) &&
        !base_in->parameters.output.do_user_defined_postprocessing)
      return;

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

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         current_time,
                                         base_in->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (base_in->parameters.output.do_user_defined_postprocessing)
      base_in->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, current_time);
  }

  template <int dim>
  void
  AdvectionDiffusionProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
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
                                 time_iterator->get_current_time_step_number());
  }

  template class AdvectionDiffusionProblem<1>;
  template class AdvectionDiffusionProblem<2>;
  template class AdvectionDiffusionProblem<3>;
} // namespace MeltPoolDG::LevelSet
