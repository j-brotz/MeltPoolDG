#include "heat_transfer_problem.hpp"
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/heat/heat_cut_operation.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/heat_diffuse_operation.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/material_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


namespace MeltPoolDG::Heat
{
  template <int dim>
  void
  HeatTransferProblem<dim>::run()
  {
    initialize();

    try
      {
        while (not time_iterator->is_finished())
          {
            const auto dt = time_iterator->compute_next_time_increment();
            simulation_case->set_time_boundary_conditions(time_iterator->get_current_time());

            time_iterator->print_me(scratch_data->get_pcout());

            if (velocity_field_function)
              compute_field_vector(velocity, velocity_dof_idx, *velocity_field_function);

            if (heaviside_field_function)
              compute_field_vector(level_set_as_heaviside,
                                   level_set_dof_idx,
                                   *heaviside_field_function);

            if (level_set_field_function)
              compute_field_vector(level_set, level_set_dof_idx, *level_set_field_function);

            // heat source
            // zero out
            heat_operation->get_heat_source() = 0;
            // add custom source field if given
            if (const auto source_field_function = simulation_case->get_field_function(
                  "prescribed_heat_source", "heat_transfer", true /*is_optional*/))
              compute_field_vector(heat_operation->get_heat_source(),
                                   temp_dof_idx,
                                   *source_field_function);
            // add laser heat source if given
            if (laser_operation)
              {
                laser_operation->move_laser(dt);

                // only precompute the laser heat source if it's not passed to the CutFEM Operator
                if (simulation_case->parameters.heat.operator_type != TwoPhaseOperatorType::cut)
                  {
                    auto heat_diffuse_operation =
                      dynamic_cast<HeatDiffuseOperation<dim> *>(heat_operation.get());
                    Assert(heat_diffuse_operation != nullptr, ExcInternalError());
                    laser_operation->compute_heat_source(heat_diffuse_operation->get_heat_source(),
                                                         heat_diffuse_operation->get_user_rhs(),
                                                         level_set_as_heaviside,
                                                         level_set_dof_idx,
                                                         temp_hanging_nodes_dof_idx,
                                                         temp_quad_idx,
                                                         false /* zero_out */);
                  }
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
            output_results(false /* output_not_converged */);

            if (simulation_case->parameters.amr.do_amr)
              refine_mesh();
          }
      }
    catch (const ExcHeatTransferNoConvergence &e)
      {
        output_results(true /* output_not_converged */);
        AssertThrow(false, e);
      }
    catch (const SolverControl::NoConvergence &e)
      {
        output_results(false /* output_not_converged */);
        AssertThrow(false, e);
      }
    Journal::print_end(scratch_data->get_pcout());
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::initialize()
  {
    const auto &param = simulation_case->parameters;
    /*
     *  setup DoFHandler
     */
    dof_handler.reinit(*simulation_case->triangulation);
    dof_handler_velocity.reinit(*simulation_case->triangulation);
    dof_handler_level_set.reinit(*simulation_case->triangulation);
    /*
     *  setup scratch data
     */
    scratch_data =
      std::make_shared<ScratchData<dim>>(simulation_case->mpi_communicator,
                                         simulation_case->parameters.base.verbosity_level,
                                         /*do_matrix_free*/ true);

    /*
     *  setup mapping
     */
    scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(param.heat.fe));

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
    temp_quad_idx =
      scratch_data->attach_quadrature(FiniteElementUtils::create_quadrature<dim>(param.heat.fe));

    /*
     *  initialize the time stepping scheme
     */
    time_iterator = std::make_shared<TimeIterator<double>>(param.time_stepping);

    /*
     * laser operation
     */
    if (param.laser.power > 0.0)
      {
        laser_operation = std::make_shared<LaserOperation<dim>>(
          *scratch_data,
          simulation_case->get_periodic_bc(),
          param.laser,
          &level_set_as_heaviside,
          level_set_dof_idx,
          &param.rad_trans,
          false,
          param.heat.operator_type != TwoPhaseOperatorType::cut,
          param.material.two_phase_fluid_properties_transition_type !=
            TwoPhaseFluidPropertiesTransitionType::sharp,
          param.output.paraview.print_boundary_id);
        laser_operation->reset(param.time_stepping.start_time);
      }

    /*
     *    set velocity field
     */
    VectorType *velocity_ptr = nullptr;
    velocity_field_function  = simulation_case->get_field_function("prescribed_velocity",
                                                                  "heat_transfer",
                                                                  true /*is_optional*/);
    if (velocity_field_function)
      velocity_ptr = &velocity;

    /*
     *    initialize the heat operation class
     */
    switch (param.heat.operator_type)
      {
          case TwoPhaseOperatorType::diffuse: {
            // set level-set as heaviside field
            VectorType *level_set_as_heaviside_ptr = nullptr;
            heaviside_field_function               = simulation_case->get_initial_condition(
              "prescribed_heaviside",
              (param.laser.model != LaserModelType::RTE) /*is_optional if laser is not RTE*/);
            if (heaviside_field_function)
              level_set_as_heaviside_ptr = &level_set_as_heaviside;

            // initialize (diffuse) material class
            material = std::make_shared<Material<double>>(
              param.material,
              determine_material_type(
                heaviside_field_function != nullptr,
                param.problem_specific_parameters.do_solidification,
                param.material.two_phase_fluid_properties_transition_type ==
                  TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation));

            heat_operation = std::make_shared<HeatDiffuseOperation<dim>>(
              *scratch_data,
              simulation_case->get_boundary_condition_manager("heat_transfer"),
              simulation_case->get_periodic_bc(),
              param.heat,
              *material,
              *time_iterator,
              temp_dof_idx,
              temp_hanging_nodes_dof_idx,
              temp_quad_idx,
              velocity_dof_idx,
              velocity_ptr,
              level_set_dof_idx,
              level_set_as_heaviside_ptr,
              param.problem_specific_parameters.do_solidification);
            break;
          }
          case TwoPhaseOperatorType::cut: {
            // set level-set field that defines the interface at the zero contour
            level_set_field_function =
              simulation_case->get_initial_condition("prescribed_signed_distance",
                                                     false /* is_optional */);

            auto heat_cut_operation = std::make_shared<HeatCutOperation<dim>>(
              *scratch_data,
              simulation_case->get_boundary_condition_manager("heat_transfer"),
              simulation_case->get_periodic_bc(),
              param.heat,
              param.material,
              param.evapor,
              *time_iterator,
              temp_dof_idx,
              temp_hanging_nodes_dof_idx,
              temp_quad_idx,
              param.problem_specific_parameters.do_solidification,
              level_set_dof_idx,
              level_set,
              velocity_dof_idx,
              velocity_ptr);

            if (laser_operation)
              heat_cut_operation->register_laser_intensity_function_and_direction(
                laser_operation->get_intensity_profile(),
                param.laser.template get_direction<dim>());

            heat_cut_operation->register_reinit_matrix_free([&](const DoFHandler<dim> &dh) {
              Assert(&dh == &scratch_data->get_dof_handler(temp_dof_idx), ExcInternalError());

              scratch_data->create_partitioning();

              heat_operation->setup_constraints(*scratch_data);

              scratch_data->build(true /*enable_boundary_faces*/,
                                  true /*enable_inner_face_loops*/,
                                  true /*enable_normal_vector_update*/);

              // recompute heat source
              scratch_data->initialize_dof_vector(heat_operation->get_heat_source(), temp_dof_idx);
              if (const auto source_field_function = simulation_case->get_field_function(
                    "prescribed_heat_source", "heat_transfer", true /*is_optional*/))
                compute_field_vector(heat_operation->get_heat_source(),
                                     temp_dof_idx,
                                     *source_field_function);
            });

            heat_operation = heat_cut_operation;
            break;
          }
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

    setup_dof_system();

    if (velocity_field_function)
      compute_field_vector(velocity, velocity_dof_idx, *velocity_field_function);
    if (heaviside_field_function)
      compute_field_vector(level_set_as_heaviside, level_set_dof_idx, *heaviside_field_function);
    if (level_set_field_function)
      compute_field_vector(level_set, level_set_dof_idx, *level_set_field_function);

    heat_operation->set_initial_condition(*simulation_case->get_initial_condition("heat_transfer"));

    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(temp_dof_idx),
                                           param.output,
                                           param.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(temp_dof_idx),
                                           scratch_data->get_pcout(1));
    /*
     *    Do initial refinement steps if requested
     */
    if (param.amr.do_amr and param.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < param.amr.n_initial_refinement_cycles; ++i)
        {
          std::ostringstream str;
          str << " #dofs T: " << heat_operation->get_temperature().size();
          Journal::print_line(scratch_data->get_pcout(), str.str(), "heat_transfer");
          refine_mesh();
          /*
           *  set initial conditions after initial AMR
           */
          heat_operation->set_initial_condition(
            *simulation_case->get_initial_condition("heat_transfer"));
        }
    /*
     *  output results of initialization
     */
    output_results(false /* output_not_converged */);
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
  HeatTransferProblem<dim>::setup_dof_system()
  {
    FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.base.fe,
                                                dof_handler_level_set);

    if (simulation_case->parameters.heat.operator_type == TwoPhaseOperatorType::cut)
      {
        // before the CutFEM operation can distribute dofs, the mesh must be classified according to
        // the level set indicator
        Assert(level_set_field_function != nullptr, ExcInternalError());
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_level_set, locally_relevant_dofs);
        level_set.reinit(dof_handler_level_set.locally_owned_dofs(),
                         locally_relevant_dofs,
                         dof_handler_level_set.get_communicator());
        level_set_field_function->set_time(time_iterator->get_current_time());
        dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                         dof_handler_level_set,
                                         *level_set_field_function,
                                         level_set);
      }

    heat_operation->distribute_dofs(dof_handler);

    FiniteElementUtils::distribute_dofs<dim, dim>(simulation_case->parameters.base.fe,
                                                  dof_handler_velocity);
    if (laser_operation)
      laser_operation->distribute_dofs(simulation_case->parameters.heat.fe);

    /*
     *  create partitioning
     */
    scratch_data->create_partitioning();
    /*
     *  create AffineConstraints
     */
    Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                        simulation_case->get_periodic_bc(),
                                        velocity_dof_idx);
    Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                        simulation_case->get_periodic_bc(),
                                        level_set_dof_idx);

    heat_operation->setup_constraints(*scratch_data);

    if (laser_operation)
      laser_operation->setup_constraints();

    {
      const bool enable_inner_face_loops =
        (simulation_case->parameters.heat.operator_type == TwoPhaseOperatorType::cut) or
        (laser_operation and
         (simulation_case->parameters.laser.model ==
            LaserModelType::interface_projection_sharp_conforming or
          simulation_case->parameters.heat.operator_type == TwoPhaseOperatorType::cut));
      const bool enable_normal_vector_update =
        simulation_case->parameters.heat.operator_type == TwoPhaseOperatorType::cut;

      scratch_data->build(true /*enable_boundary_face_loops*/,
                          enable_inner_face_loops,
                          enable_normal_vector_update);
    }

    heat_operation->reinit();
    if (laser_operation)
      laser_operation->reinit();
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::output_results(const bool output_not_converged)
  {
    const unsigned int n_time_step = time_iterator->get_current_time_step_number();
    const double       time        = time_iterator->get_current_time();

    if (not post_processor->is_output_timestep(n_time_step, time) and not output_not_converged and
        not simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         time,
                                         simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    if (output_not_converged)
      heat_operation->attach_output_vectors_failed_step(generic_data_out);

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

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
    /**
     *  prescribed level set
     */
    if (level_set_field_function)
      data_out.add_data_vector(dof_handler_level_set, level_set, "level_set");


    if (laser_operation)
      laser_operation->attach_output_vectors(data_out);
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::refine_mesh()
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(simulation_case->triangulation->n_active_cells());

      switch (simulation_case->parameters.problem_specific_parameters.amr_strategy)
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
        simulation_case->parameters.amr.upper_perc_to_refine,
        simulation_case->parameters.amr.lower_perc_to_coarsen);

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
      if (level_set_field_function)
        compute_field_vector(level_set, level_set_dof_idx, *level_set_field_function);
      if (laser_operation)
        laser_operation->distribute_constraints();
    };

    const auto setup_dof_system = [&]() { this->setup_dof_system(); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 simulation_case->parameters.amr,
                                 *simulation_case->triangulation,
                                 time_iterator->get_current_time_step_number());
  }

  template class HeatTransferProblem<1>;
  template class HeatTransferProblem<2>;
  template class HeatTransferProblem<3>;
} // namespace MeltPoolDG::Heat


int
main(int argc, char *argv[])
{
  using namespace MeltPoolDG;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;
  // check command line arguments
  if (argc == 1)
    {
      if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        std::cout << "ERROR: No .json parameter files has been provided!" << std::endl;
      return 1;
    }
  else if (argc == 2)
    {
      input_file = std::string(argv[argc - 1]);
      run_simulation<Heat::HeatTransferCaseParameters<double>,
                     Heat::HeatTransferCase,
                     Heat::HeatTransferProblem>(input_file, mpi_comm);
    }
  else if (argc == 3 and
           (std::string(argv[1]) == "--help" or std::string(argv[1]) == "--help-detail"))
    {
      input_file = std::string(argv[argc - 1]);

      dealii::ParameterHandler                 prm;
      Heat::HeatTransferCaseParameters<double> parameters;
      parameters.process_parameters_file(prm, input_file);

      if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        parameters.print_parameters(prm,
                                    std::cout,
                                    std::string(argv[1]) == "--help-detail" /*print_details*/);

      return 0;
    }
  else
    AssertThrow(false, dealii::ExcMessage("no input file specified"));

  return 0;
}
//