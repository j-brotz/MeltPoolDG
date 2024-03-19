#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/interface/problem_base.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_problem.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  void
  LevelSetProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);
    ScopedName         sc("run");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);

    while (!time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout());

        //@todo: adapt in case of adaptive time stepping
        if (profiling_monitor && profiling_monitor->now())
          {
            profiling_monitor->print(scratch_data->get_pcout(),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }
        compute_advection_velocity(*base_in->get_advection_field("level_set"));

        if (evaporation_operation)
          {
            // Only if a spatially constant evaporative mass flux is given as an analytical
            // function, the time is needed to evaluate the function.
            if (base_in->parameters.evapor.evaporative_mass_flux_model ==
                Evaporation::EvaporationModelType::analytical)
              evaporation_operation->set_time(time_iterator->get_current_time());
            else
              AssertThrow(false,
                          ExcMessage("Only a evaporation model of constant type is supported."));

            // compute evaporative mass flux spatially constant
            evaporation_operation->compute_evaporative_mass_flux();

            // compute velocity due to evaporative mass flux
            evaporation_operation->compute_evaporation_velocity();

            // compute advection velocity of the interface
            advection_velocity += evaporation_operation->get_velocity();
          }

        level_set_operation->solve();


        // do output if requested
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time(),
                       base_in);

        if (base_in->parameters.amr.do_amr)
          refine_mesh(base_in);
      }

    Journal::print_end(scratch_data->get_pcout());
    //... always print timing statistics
    if (profiling_monitor)
      {
        profiling_monitor->print(scratch_data->get_pcout(),
                                 scratch_data->get_timer(),
                                 scratch_data->get_mpi_comm());
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
    ScopedName         sc("initialize");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);
    /*
     *  setup mapping
     */
    scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(base_in->parameters.ls.fe));
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
    ls_quad_idx = scratch_data->attach_quadrature(
      FiniteElementUtils::create_quadrature<dim>(base_in->parameters.ls.fe));
    /*
     *  initialize the time iterator
     */
    time_iterator = std::make_shared<TimeIterator<double>>(base_in->parameters.time_stepping);

    setup_dof_system(base_in, false);

    /*
     * initialize the levelset operation class
     */

    level_set_operation = std::make_shared<LevelSetOperation<dim>>(*scratch_data,
                                                                   *time_iterator,
                                                                   base_in,
                                                                   advection_velocity,
                                                                   ls_dof_idx,
                                                                   ls_hanging_nodes_dof_idx,
                                                                   ls_quad_idx,
                                                                   reinit_dof_idx,
                                                                   curv_dof_idx,
                                                                   normal_dof_idx,
                                                                   vel_dof_idx,
                                                                   ls_zero_bc_idx);
    level_set_operation->reinit();

    /*
     * set initial conditions
     */
    compute_advection_velocity(*base_in->get_advection_field("level_set"));

    /*
     * configure level set with evaporation if requested
     */
    if (base_in->parameters.base.problem_name == ProblemType::level_set_with_evaporation)
      {
        evaporation_operation = std::make_shared<Evaporation::EvaporationOperation<dim>>(
          *scratch_data,
          level_set_operation->get_level_set_as_heaviside(),
          level_set_operation->get_normal_vector(),
          base_in->parameters.evapor,
          base_in->parameters.material,
          normal_dof_idx,
          vel_dof_idx,
          ls_hanging_nodes_dof_idx,
          ls_hanging_nodes_dof_idx,
          ls_quad_idx);
      }
    /*
     *  set initial conditions of the level set field ...
     */
    if (const auto initial_field =
          base_in->get_initial_condition("level_set", true /*is optional*/))
      {
        // ... via a given level set field
        level_set_operation->set_initial_condition(*initial_field);
      }
    else if (const auto initial_field =
               base_in->get_initial_condition("signed_distance", true /*is optional*/))
      {
        // ... or a given signed distance field.
        level_set_operation->set_initial_condition(*initial_field,
                                                   true /*is signed distance function*/);
      }
    else
      AssertThrow(
        false,
        ExcMessage("For the level set operation either a function for the initial level set or the "
                   "signed distance field must be provided. Abort ..."));
    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(ls_dof_idx),
                                           base_in->parameters.output,
                                           base_in->parameters.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(ls_dof_idx),
                                           scratch_data->get_pcout(1));
    /*
     *  initialize profiling
     */
    if (base_in->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<double>>(base_in->parameters.profiling,
                                                              *time_iterator);
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
          if (const auto initial_field =
                base_in->get_initial_condition("level_set", true /*is optional*/))
            {
              // ... via a given level set field
              level_set_operation->set_initial_condition(*initial_field);
            }
          else if (const auto initial_field =
                     base_in->get_initial_condition("signed_distance", true /*is optional*/))
            {
              // ... or a given signed distance field.
              level_set_operation->set_initial_condition(*initial_field,
                                                         true /*is signed distance function*/);
            }
          else
            AssertThrow(
              false,
              ExcMessage(
                "For the level set operation either a function for the initial level set or the "
                "signed distance field must be provided. Abort ..."));
        }

    output_results(0, base_in->parameters.time_stepping.start_time, base_in);
  }

  template <int dim>
  void
  LevelSetProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                         const bool                           do_reinit)
  {
    FiniteElementUtils::distribute_dofs<dim, 1>(base_in->parameters.ls.fe, dof_handler);
    FiniteElementUtils::distribute_dofs<dim, dim>(base_in->parameters.base.fe,
                                                  dof_handler_velocity);
    /*
     *  create partitioning
     */
    scratch_data->create_partitioning();
    /*
     *  create AffineConstraints
     */
    base_in->attach_boundary_condition("level_set"); //@todo move to a more central place
    MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_BC_into_DBC<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("level_set"),
      base_in->get_periodic_bc(),
      ls_dof_idx,
      ls_hanging_nodes_dof_idx);

    MeltPoolDG::Constraints::make_DBC_and_HNC_and_merge_HNC_into_DBC<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("level_set"),
      ls_zero_bc_idx,
      ls_hanging_nodes_dof_idx,
      false /*set inhomogeneities to zero*/);

    MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                    base_in->get_periodic_bc(),
                                                    vel_dof_idx);
    /*
     *  create the matrix-free object
     */
    scratch_data->build(false, false);

    if (do_reinit)
      {
        level_set_operation->reinit();
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
    advec_func.set_time(time_iterator->get_current_time());

    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     dof_handler_velocity,
                                     advec_func,
                                     advection_velocity);
    hanging_node_constraints_velocity.distribute(advection_velocity);
  }
  /*
   *  This function is to create output
   */
  template <int dim>
  void
  LevelSetProblem<dim>::output_results(const unsigned int                   time_step,
                                       const double                         time,
                                       std::shared_ptr<SimulationBase<dim>> base_in)
  {
    ScopedName         sc("output_results");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);
    if (!post_processor->is_output_timestep(time_step, time))
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      level_set_operation->attach_output_vectors(data_out);
      if (evaporation_operation)
        evaporation_operation->attach_output_vectors(data_out);

      /*
       *  output advection velocity
       */
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(dof_handler_velocity,
                               advection_velocity,
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         time,
                                         base_in->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (base_in->parameters.output.do_user_defined_postprocessing)
      base_in->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, time);
  }

  /*
   *  perform mesh refinement
   */
  template <int dim>
  void
  LevelSetProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    ScopedName         sc("AMR");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);
    const auto         mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      if (problem_specific_parameters.amr.strategy == AMRStrategy::generic)
        {
          Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

          VectorType locally_relevant_solution;
          locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));
          locally_relevant_solution.copy_locally_owned_data_from(
            level_set_operation->get_level_set());
          constraints_dirichlet.distribute(locally_relevant_solution);
          locally_relevant_solution.update_ghost_values();

          for (unsigned int i = 0; i < locally_relevant_solution.locally_owned_size(); ++i)
            locally_relevant_solution.local_element(i) =
              (1.0 - locally_relevant_solution.local_element(i) *
                       locally_relevant_solution.local_element(i));

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
        }
      else if (problem_specific_parameters.amr.strategy == AMRStrategy::refine_all_interface_cells)
        {
          const bool solution_update_ghosts =
            !level_set_operation->get_level_set().has_ghost_elements();
          if (solution_update_ghosts)
            level_set_operation->get_level_set().update_ghost_values();
          const bool hs_update_ghosts =
            !level_set_operation->get_level_set_as_heaviside().has_ghost_elements();
          if (hs_update_ghosts)
            level_set_operation->get_level_set_as_heaviside().update_ghost_values();
          FEValues<dim>       ls_values(scratch_data->get_mapping(),
                                  scratch_data->get_fe(ls_dof_idx),
                                  Quadrature<dim>(
                                    scratch_data->get_fe(ls_dof_idx).get_unit_support_points()),
                                  update_values);
          std::vector<double> ls_vals(scratch_data->get_n_dofs_per_cell(ls_dof_idx));
          std::vector<double> hs_vals(scratch_data->get_n_dofs_per_cell(ls_dof_idx));

          const double phi_threshold = std::tanh(2.0);

          for (auto &cell : tria.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  TriaIterator<DoFCellAccessor<dim, dim, false>> ls_dof_cell(
                    &tria,
                    cell->level(),
                    cell->index(),
                    &scratch_data->get_dof_handler(ls_dof_idx));
                  ls_values.reinit(ls_dof_cell);

                  ls_values.get_function_values(level_set_operation->get_level_set(), ls_vals);
                  ls_values.get_function_values(level_set_operation->get_level_set_as_heaviside(),
                                                hs_vals);

                  double max_ls = 0;
                  for (const auto &ls : ls_vals)
                    max_ls = std::max(std::abs(ls), max_ls);

                  bool refine_cell =
                    (cell->level() <
                       static_cast<int>(base_in->parameters.amr.max_grid_refinement_level) &&
                     (max_ls < phi_threshold));

                  if (refine_cell == true)
                    cell->set_refine_flag();
                  else if ((cell->level() >= base_in->parameters.amr.min_grid_refinement_level) &&
                           (max_ls >= phi_threshold))
                    cell->set_coarsen_flag();
                }
            }

          if (solution_update_ghosts)
            level_set_operation->get_level_set().zero_out_ghost_values();
          if (hs_update_ghosts)
            level_set_operation->get_level_set_as_heaviside().zero_out_ghost_values();
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      level_set_operation->attach_vectors(vectors);
      if (evaporation_operation)
        evaporation_operation->attach_vectors(vectors);
    };

    const auto post = [&]() {
      constraints_dirichlet.distribute(level_set_operation->get_level_set());
      hanging_node_constraints.distribute(level_set_operation->get_level_set_as_heaviside());
    };

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 base_in->parameters.amr,
                                 dof_handler,
                                 time_iterator->get_current_time_step_number());
  }

  template <int dim>
  void
  LevelSetProblem<dim>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("problem specific");
    {
      prm.enter_subsection("amr");
      {
        prm.add_parameter("strategy",
                          problem_specific_parameters.amr.strategy,
                          "Select the AMR strategy.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }


  template class LevelSetProblem<1>;
  template class LevelSetProblem<2>;
  template class LevelSetProblem<3>;
} // namespace MeltPoolDG::LevelSet
