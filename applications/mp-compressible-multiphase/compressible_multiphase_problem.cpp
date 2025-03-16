#include "compressible_multiphase_problem.hpp"

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/flow/compressible_flow_boundary_conditions.hpp>
#include <meltpooldg/flow/compressible_multiphase/compressible_multiphase_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include "compressible_multiphase_case.hpp"

namespace MeltPoolDG::Multiphase
{
  template <int dim>
  void
  CompressibleMultiphaseProblem<dim>::run()
  {
    initialize();

    // output initial condition
    output_results(time_iterator->get_current_time_step_number(),
                   time_iterator->get_current_time());

    while (!time_iterator->is_finished())
      {
        // use CFL condition to compute time step size if required
        if (simulation_case->parameters.flow.do_cfl_time_stepping)
          time_iterator->set_current_time_increment(
            comp_multiphase_operation.compute_time_step_size(), std::numeric_limits<double>::max());

        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));

        // do level-set advection and reinitialization
        update_level_set();

        comp_multiphase_operation.solve(time_iterator->get_current_time(),
                                        time_iterator->get_current_time_increment());

        // do output if requested
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

        if (profiling_monitor and profiling_monitor->now())
          {
            profiling_monitor->print(scratch_data->get_pcout(1),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }
      }

    // always print timing statistics
    if (profiling_monitor)
      {
        profiling_monitor->print(scratch_data->get_pcout(1),
                                 scratch_data->get_timer(),
                                 scratch_data->get_mpi_comm());
      }
    Journal::print_end(scratch_data->get_pcout(1));
  }

  template <int dim>
  void
  CompressibleMultiphaseProblem<dim>::interpolate_initial_level_set()
  {
    // set the current time to the field function
    level_set_field_function->set_time(time_iterator->get_current_time());

    // interpolate the values of the given field function to the dof vector values
    level_set.zero_out_ghost_values();
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(level_set_dof_idx),
                                     *level_set_field_function,
                                     level_set);
    level_set.update_ghost_values();
  }

  template <int dim>
  void
  CompressibleMultiphaseProblem<dim>::update_level_set()
  {
    // TODO: introduce dof_handler for level-set advection velocity field, compute projection of
    // interface velocity in normal direction of the interface to reduce the distortion of the
    // level-set field
    // TODO: use advection_DG_operation and reinitialization_..._operation for dim>2 cases with
    // distorted level-set Currently, we only consider static interfaces
  }

  template <int dim>
  void
  CompressibleMultiphaseProblem<dim>::setup_dof_system()
  {
    comp_multiphase_operation.distribute_dofs(dof_handler);

    scratch_data->create_partitioning();

    simulation_case->set_time_boundary_conditions(
      simulation_case->parameters.time_stepping.start_time);

    // create the matrix-free object
    scratch_data->build(true, true, true, simulation_case->parameters.flow.fe.degree == 2);

    // Currently, only homogeneous Cartesian grids without mesh refinements are enabled for
    // cutDG
    AssertThrow(
      std::abs(scratch_data->get_min_cell_size() - scratch_data->get_max_cell_size()) < 1e-10,
      dealii::ExcMessage(
        "Only homogeneous Cartesian grids without local grid refinements are supported!"));

    // print mesh information
    {
      ScopedName sc("compFlow::cells");
      CellMonitor::add_info(sc,
                            scratch_data->get_triangulation().n_global_active_cells(),
                            scratch_data->get_min_cell_size(),
                            scratch_data->get_max_cell_size());
    }
  }

  template <int dim>
  void
  CompressibleMultiphaseProblem<dim>::initialize()
  {
    // setup DoFHandler
    dof_handler.reinit(*simulation_case->triangulation);
    dof_handler_level_set.reinit(*simulation_case->triangulation);

    // setup scratch data
    {
      scratch_data =
        std::make_shared<ScratchData<dim>>(simulation_case->mpi_communicator,
                                           simulation_case->parameters.base.verbosity_level,
                                           true);
      // set up mapping
      scratch_data->set_mapping(
        FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.flow.fe));

      // create quadrature rule
      comp_multiphase_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.flow.fe));

      comp_multiphase_dof_idx = scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_constraint_matrix(constraints);

      level_set_dof_idx = scratch_data->attach_dof_handler(dof_handler_level_set);
      scratch_data->attach_constraint_matrix(constraints_level_set);
    }

    // set analytical level-set field function
    level_set_field_function = simulation_case->get_field_function("level_set",
                                                                   "compressible_multiphase",
                                                                   false /* is_optional */);

    // initialize the time iterator
    time_iterator =
      std::make_shared<TimeIterator<double>>(simulation_case->parameters.time_stepping);

    // initialize compressible multiphase operation
    std::unique_ptr<CompressibleMultiphaseOperation<dim, double>> operation =
      std::make_unique<CompressibleMultiphaseOperation<dim, double>>(
        *scratch_data,
        simulation_case->parameters.flow,
        *time_iterator,
        [&](const dealii::DoFHandler<dim> &dh) {
          Assert(&dh == &scratch_data->get_dof_handler(comp_multiphase_dof_idx),
                 dealii::ExcInternalError());

          scratch_data->create_partitioning();

          // create the matrix-free object
          scratch_data->build(true, true, true, simulation_case->parameters.flow.fe.degree == 2);

          comp_multiphase_operation.reinit();
        },
        comp_multiphase_dof_idx,
        level_set_dof_idx,
        comp_multiphase_quad_idx,
        level_set);

    comp_multiphase_operation =
      MeltPoolDG::Flow::CompressibleFlowOperation<dim, double>(std::move(operation));

    // set up level-set for cutDG
    // currently, we use a continuous level-set field with same element degree as the flow field
    const FE_Q<dim> fe_level_set(simulation_case->parameters.flow.fe.degree);
    dof_handler_level_set.distribute_dofs(fe_level_set);

    Assert(level_set_field_function != nullptr, ExcInternalError());

    level_set.reinit(dof_handler_level_set.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler_level_set),
                     dof_handler_level_set.get_communicator());
    interpolate_initial_level_set();

    setup_dof_system();

    // set boundary conditions
    comp_multiphase_operation.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::inflow,
      simulation_case->get_boundary_condition("inflow", "compressible_multiphase"));

    comp_multiphase_operation.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::subsonic_outflow_fixed_pressure,
      simulation_case->get_boundary_condition("outflow_fixed_pressure", "compressible_multiphase"));

    comp_multiphase_operation.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::subsonic_outflow_fixed_energy,
      simulation_case->get_boundary_condition("outflow_fixed_energy", "compressible_multiphase"));

    comp_multiphase_operation.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::slip_wall,
      simulation_case->get_boundary_condition("slip_wall", "compressible_multiphase"));

    comp_multiphase_operation.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::no_slip_wall,
      simulation_case->get_boundary_condition("no_slip_wall", "compressible_multiphase"));

    // initialize operation
    comp_multiphase_operation.reinit();

    // set initial condition for the flow field
    comp_multiphase_operation.set_initial_condition(
      *simulation_case->get_initial_condition("compressible_multiphase"));

    // set body force
    if (dim > 1 and simulation_case->parameters.flow.gravity_constant > 0.)
      {
        std::unique_ptr<Functions::ConstantFunction<dim>> body_force =
          std::make_unique<Functions::ConstantFunction<dim>>(
            dim > 2 ?
              std::vector<double>({0., 0., -simulation_case->parameters.flow.gravity_constant}) :
              std::vector<double>({0., -simulation_case->parameters.flow.gravity_constant}));
        comp_multiphase_operation.set_body_force(std::move(body_force));
      }

    // initialize postprocessor
    post_processor =
      std::make_unique<Postprocessor<dim>>(scratch_data->get_mpi_comm(comp_multiphase_dof_idx),
                                           simulation_case->parameters.output,
                                           simulation_case->parameters.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(comp_multiphase_dof_idx),
                                           scratch_data->get_pcout(2));

    // initialize profiling
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<double>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim>
  void
  CompressibleMultiphaseProblem<dim>::output_results(const unsigned int time_step,
                                                     const double       current_time)
  {
    if (not post_processor->is_output_timestep(time_step, current_time) and
        not simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      comp_multiphase_operation.attach_output_vectors(data_out);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         current_time,
                                         simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    generic_data_out.add_data_vector(dof_handler_level_set, level_set, "level_set");

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing and
        post_processor->is_output_timestep(time_step, current_time))
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    // TODO: build_patches does not work yet
    // post_processor->process(time_step, generic_data_out, current_time);

    DataOut<dim> data_out;

    if (dim == 2)
      {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out.set_flags(flags);
      }

    data_out.add_data_vector(dof_handler_level_set, level_set, "level_set");
    data_out.add_data_vector(dof_handler, comp_multiphase_operation.get_solution(), "solution");

    // finalize
    data_out.build_patches();
    const std::string filename = simulation_case->parameters.output.paraview.filename +
                                 Utilities::int_to_string(result_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, scratch_data->get_triangulation().get_communicator());
    result_number++;
  }

  template class CompressibleMultiphaseProblem<1>;
  template class CompressibleMultiphaseProblem<2>;
  template class CompressibleMultiphaseProblem<3>;

} // namespace MeltPoolDG::Multiphase

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::Multiphase::CompressibleMultiphaseCaseParameters<double>,
                           MeltPoolDG::Multiphase::CompressibleMultiphaseCase,
                           MeltPoolDG::Multiphase::CompressibleMultiphaseProblem>(argc,
                                                                                  argv,
                                                                                  mpi_comm);
  return 0;
}
