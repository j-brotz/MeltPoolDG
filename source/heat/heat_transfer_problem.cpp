#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <meltpooldg/heat/heat_transfer_problem.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <functional>
#include <sstream>
#include <vector>

namespace MeltPoolDG::Heat
{
  template <int dim>
  void
  HeatTransferProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    try
      {
        while (!time_iterator->is_finished())
          {
            const auto dt = time_iterator->compute_next_time_increment();

            time_iterator->print_me(scratch_data->get_pcout());

            if (velocity_field_function)
              compute_field_vector(velocity, velocity_dof_idx, *velocity_field_function);

            if (heaviside_field_function)
              compute_field_vector(level_set_as_heaviside,
                                   level_set_dof_idx,
                                   *heaviside_field_function);

            // heat source
            // zero out
            heat_operation->get_heat_source() = 0;
            // add custom source field if given
            if (const auto source_field_function =
                  base_in->get_source_field("heat_transfer", true /*is_optional*/))
              compute_field_vector(heat_operation->get_heat_source(),
                                   temp_dof_idx,
                                   *source_field_function);
            // add laser heat source if given
            if (laser_operation)
              {
                laser_operation->move_laser(dt);
                laser_operation->compute_heat_source(heat_operation->get_heat_source(),
                                                     heat_operation->get_user_rhs(),
                                                     level_set_as_heaviside,
                                                     level_set_dof_idx,
                                                     temp_hanging_nodes_dof_idx,
                                                     temp_quad_idx,
                                                     false /* zero_out */);
              }

            Journal::print_formatted_norm(
              scratch_data->get_pcout(2),
              [&]() -> double {
                return VectorTools::compute_norm(heat_operation->get_heat_source(),
                                                 *scratch_data,
                                                 temp_dof_idx,
                                                 temp_quad_idx,
                                                 dealii::VectorTools::NormType::L1_norm);
              },
              "heat-source",
              "laser",
              11 /*precision*/,
              "L1");

            heat_operation->solve();

            // ... and output the results to vtk files.
            output_results(base_in, false);

            if (base_in->parameters.amr.do_amr)
              refine_mesh(base_in);
          }
      }
    catch (const ExcHeatTransferNoConvergence &e)
      {
        output_results(base_in, true);
        AssertThrow(false, e);
      }
    catch (const SolverControl::NoConvergence &e)
      {
        output_results(base_in, false);
        AssertThrow(false, e);
      }
    Journal::print_end(scratch_data->get_pcout());
  }

  template <int dim>
  std::string
  HeatTransferProblem<dim>::get_name()
  {
    return "heat_transfer";
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("problem specific");
    {
      prm.add_parameter(
        "do solidification",
        problem_specific_parameters.do_solidification,
        "Set this parameter to true if you want to consider melting/solidification effects.");
      prm.add_parameter("amr strategy",
                        problem_specific_parameters.amr_strategy,
                        "Select the AMR strategy.");
    }
    prm.leave_subsection();
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  Add problem specific parameters defined within add_parameters()
     */
    this->add_problem_specific_parameters(base_in->parameter_file);
    /*
     *  setup DoFHandler
     */
    dof_handler.reinit(*base_in->triangulation);
    dof_handler_velocity.reinit(*base_in->triangulation);
    dof_handler_level_set.reinit(*base_in->triangulation);
    /*
     *  setup scratch data
     */
    scratch_data = std::make_shared<ScratchData<dim>>(base_in->mpi_communicator,
                                                      base_in->parameters.base.verbosity_level,
                                                      /*do_matrix_free*/ true);

    /*
     *  setup mapping
     */
    scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(base_in->parameters.heat.fe));

    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler_velocity);
    scratch_data->attach_dof_handler(dof_handler_level_set);

    /*
     * attach constraints
     */
    temp_dof_idx = scratch_data->attach_constraint_matrix(temp_constraints);
    temp_hanging_nodes_dof_idx =
      scratch_data->attach_constraint_matrix(temp_hanging_nodes_constraints);
    velocity_dof_idx  = scratch_data->attach_constraint_matrix(velocity_hanging_nodes_constraints);
    level_set_dof_idx = scratch_data->attach_constraint_matrix(level_set_hanging_nodes_constraints);

    /*
     *  create quadrature rule
     */
    temp_quad_idx = scratch_data->attach_quadrature(
      FiniteElementUtils::create_quadrature<dim>(base_in->parameters.heat.fe));

    /*
     * laser operation
     */
    if (base_in->parameters.laser.power > 0.0)
      {
        laser_operation = std::make_shared<LaserOperation<dim>>(*scratch_data,
                                                                base_in->get_periodic_bc(),
                                                                base_in->parameters,
                                                                &level_set_as_heaviside,
                                                                level_set_dof_idx);
        laser_operation->reset(base_in->parameters.time_stepping.start_time);
      }

    setup_dof_system(base_in, false);

    if (laser_operation)
      laser_operation->reinit();


    /*
     *  initialize the time stepping scheme
     */
    time_iterator = std::make_shared<TimeIterator<double>>(base_in->parameters.time_stepping);
    /*
     *    set velocity field
     */
    VectorType *velocity_ptr = nullptr;
    velocity_field_function  = base_in->get_velocity_field("heat_transfer", true /*is_optional*/);
    if (velocity_field_function)
      {
        compute_field_vector(velocity, velocity_dof_idx, *velocity_field_function);
        velocity_ptr = &velocity;
      }
    /*
     *    set level-set as heaviside field
     */
    VectorType *level_set_as_heaviside_ptr = nullptr;
    heaviside_field_function =
      base_in->get_initial_condition("prescribed_heaviside",
                                     (base_in->parameters.laser.model !=
                                      LaserModelType::RTE) /*is_optional if laser is not RTE*/);
    if (heaviside_field_function)
      {
        compute_field_vector(level_set_as_heaviside, level_set_dof_idx, *heaviside_field_function);
        level_set_as_heaviside_ptr = &level_set_as_heaviside;
      }
    /*
     * initialize material
     */
    const auto material_type = determine_material_type(
      heaviside_field_function != nullptr,
      problem_specific_parameters.do_solidification,
      base_in->parameters.material.two_phase_fluid_properties_transition_type ==
        TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation);
    material = std::make_shared<Material<double>>(base_in->parameters.material, material_type);
    /*
     *    initialize the heat operation class
     */
    heat_operation =
      std::make_shared<HeatTransferOperation<dim>>(base_in->get_bc("heat_transfer"),
                                                   *scratch_data,
                                                   base_in->parameters.heat,
                                                   *material,
                                                   *time_iterator,
                                                   temp_dof_idx,
                                                   temp_hanging_nodes_dof_idx,
                                                   temp_quad_idx,
                                                   velocity_dof_idx,
                                                   velocity_ptr,
                                                   level_set_dof_idx,
                                                   level_set_as_heaviside_ptr,
                                                   problem_specific_parameters.do_solidification);

    heat_operation->set_initial_condition(*base_in->get_initial_condition("heat_transfer"),
                                          base_in->parameters.time_stepping.start_time);

    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(temp_dof_idx),
                                           base_in->parameters.output,
                                           base_in->parameters.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(temp_dof_idx),
                                           scratch_data->get_pcout(1));
    /*
     *    Do initial refinement steps if requested
     */
    if (base_in->parameters.amr.do_amr && base_in->parameters.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
        {
          std::ostringstream str;
          str << " #dofs T: " << heat_operation->get_temperature().size();
          Journal::print_line(scratch_data->get_pcout(), str.str(), "heat_transfer");
          refine_mesh(base_in);
          /*
           *  set initial conditions after initial AMR
           */
          heat_operation->set_initial_condition(*base_in->get_initial_condition("heat_transfer"),
                                                base_in->parameters.time_stepping.start_time);
        }
    /*
     *  output results of initialization
     */
    output_results(base_in);
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::compute_field_vector(VectorType        &vector,
                                                 const unsigned int dof_idx,
                                                 Function<dim>     &field_function)
  {
    scratch_data->initialize_dof_vector(vector, dof_idx);
    /*
     *  set the current time to the advection field function
     */
    field_function.set_time(time_iterator->get_current_time());
    /*
     *  interpolate the values of the advection velocity
     */
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(dof_idx),
                                     field_function,
                                     vector);
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                             const bool                           do_reinit)
  {
    FiniteElementUtils::distribute_dofs<dim, 1>(base_in->parameters.heat.fe, dof_handler);
    FiniteElementUtils::distribute_dofs<dim, dim>(base_in->parameters.base.fe,
                                                  dof_handler_velocity);
    FiniteElementUtils::distribute_dofs<dim, 1>(base_in->parameters.base.fe, dof_handler_level_set);
    if (laser_operation)
      laser_operation->distribute_dofs(base_in->parameters.heat.fe);

    /*
     *  create partitioning
     */
    scratch_data->create_partitioning();
    /*
     *  create AffineConstraints
     */
    MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                    base_in->get_periodic_bc(),
                                                    velocity_dof_idx);
    MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                    base_in->get_periodic_bc(),
                                                    level_set_dof_idx);

    base_in->attach_boundary_condition("heat_transfer"); //@todo move to a more central place
    MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_BC_into_DBC<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("heat_transfer"),
      base_in->get_periodic_bc(),
      temp_dof_idx,
      temp_hanging_nodes_dof_idx);

    if (laser_operation)
      laser_operation->setup_constraints();

    scratch_data->build(
      true /*enable_boundary_faces*/,
      base_in->parameters.laser.model ==
        LaserModelType::interface_projection_sharp_conforming /*enable_inner_face_loops*/);

    if (do_reinit)
      {
        heat_operation->reinit();
        if (laser_operation)
          laser_operation->reinit();
      }
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::output_results(std::shared_ptr<SimulationBase<dim>> base_in,
                                           const bool output_not_converged)
  {
    const unsigned int n_time_step = time_iterator->get_current_time_step_number();
    const double       time        = time_iterator->get_current_time();

    if (!post_processor->is_output_timestep(n_time_step, time) && !output_not_converged &&
        !base_in->parameters.output.do_user_defined_postprocessing)
      return;

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         time,
                                         base_in->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    if (output_not_converged)
      heat_operation->attach_output_vectors_failed_step(generic_data_out);

    // user-defined postprocessing
    if (base_in->parameters.output.do_user_defined_postprocessing)
      base_in->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(n_time_step,
                            generic_data_out,
                            time,
                            output_not_converged /* force_output */,
                            output_not_converged /* force_update_requested_output_variables */);
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    heat_operation->attach_output_vectors(data_out);

    /**
     *  prescribed velocity
     */
    if (velocity_field_function)
      {
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vector_component_interpretation(dim,
                                          DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(dof_handler_velocity,
                                 velocity,
                                 std::vector<std::string>(dim, "velocity"),
                                 vector_component_interpretation);
      }
    /**
     *  prescribed heaviside
     */
    if (heaviside_field_function)
      data_out.add_data_vector(dof_handler_level_set, level_set_as_heaviside, "heaviside");


    if (laser_operation)
      laser_operation->attach_output_vectors(data_out);
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

      switch (problem_specific_parameters.amr_strategy)
        {
            case AMRStrategy::KellyErrorEstimator: {
              VectorType locally_relevant_solution;
              locally_relevant_solution.reinit(scratch_data->get_partitioner(temp_dof_idx));
              locally_relevant_solution.copy_locally_owned_data_from(
                heat_operation->get_temperature());
              scratch_data->get_constraint(temp_dof_idx).distribute(locally_relevant_solution);
              locally_relevant_solution.update_ghost_values();

              KellyErrorEstimator<dim>::estimate(scratch_data->get_mapping(),
                                                 scratch_data->get_dof_handler(temp_dof_idx),
                                                 scratch_data->get_face_quadrature(temp_quad_idx),
                                                 {}, // neumann bc
                                                 locally_relevant_solution,
                                                 estimated_error_per_cell);
              break;
            }
            case AMRStrategy::generic: {
              VectorType locally_relevant_solution;
              locally_relevant_solution.reinit(scratch_data->get_partitioner(level_set_dof_idx));

              locally_relevant_solution.copy_locally_owned_data_from(level_set_as_heaviside);

              for (unsigned int i = 0; i < locally_relevant_solution.locally_owned_size(); ++i)
                locally_relevant_solution.local_element(i) =
                  (1.0 - Utilities::fixed_power<2>(
                           2.0 * locally_relevant_solution.local_element(i) - 1.0));

              locally_relevant_solution.update_ghost_values();

              dealii::VectorTools::integrate_difference(dof_handler_level_set,
                                                        locally_relevant_solution,
                                                        Functions::ZeroFunction<dim>(),
                                                        estimated_error_per_cell,
                                                        scratch_data->get_quadrature(temp_dof_idx),
                                                        dealii::VectorTools::L2_norm);
              break;
            }
          default:
            AssertThrow(false, ExcNotImplemented());
        }

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        base_in->parameters.amr.upper_perc_to_refine,
        base_in->parameters.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors =
      [&](std::vector<std::pair<const DoFHandler<dim> *,
                                std::function<void(std::vector<VectorType *> &)>>> &data) {
        data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
          heat_operation->attach_vectors(vectors);
        });
        if (laser_operation)
          laser_operation->attach_vectors(data);
      };

    const auto post = [&]() {
      heat_operation->distribute_constraints();

      if (velocity_field_function)
        compute_field_vector(velocity, velocity_dof_idx, *velocity_field_function);
      if (heaviside_field_function)
        compute_field_vector(level_set_as_heaviside, level_set_dof_idx, *heaviside_field_function);
      if (laser_operation)
        laser_operation->distribute_constraints();
    };

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in, true); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 base_in->parameters.amr,
                                 *base_in->triangulation,
                                 time_iterator->get_current_time_step_number());
  }

  template class HeatTransferProblem<1>;
  template class HeatTransferProblem<2>;
  template class HeatTransferProblem<3>;
} // namespace MeltPoolDG::Heat
