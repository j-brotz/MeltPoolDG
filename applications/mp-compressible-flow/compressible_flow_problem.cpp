#include "compressible_flow_problem.hpp"

#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include "compressible_flow_case.hpp"

namespace MeltPoolDG::Flow
{
  template <int dim>
  void
  CompressibleFlowProblem<dim>::run()
  {
    initialize();

    while (!time_iterator->is_finished())
      {
        // use CFL condition to compute time step size if required
        if (simulation_case->parameters.flow.do_cfl_time_stepping)
          time_iterator->set_current_time_increment(comp_flow_operation->compute_time_step_size(),
                                                    std::numeric_limits<double>::max());

        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout());

        comp_flow_operation->solve(time_iterator->get_current_time(),
                                   time_iterator->get_current_time_increment());

        // do output if requested
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

        if (profiling_monitor && profiling_monitor->now())
          {
            profiling_monitor->print(scratch_data->get_pcout(),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }
      }

    // always print timing statistics
    if (profiling_monitor)
      {
        profiling_monitor->print(scratch_data->get_pcout(),
                                 scratch_data->get_timer(),
                                 scratch_data->get_mpi_comm());
      }
    Journal::print_end(scratch_data->get_pcout());
  }

  template <int dim>
  void
  CompressibleFlowProblem<dim>::setup_dof_system()
  {
    // setup DoFHandler
    constexpr unsigned int dofs_per_node = dim + 2;
    FiniteElementUtils::distribute_dofs<dim, dofs_per_node>(simulation_case->parameters.flow.fe,
                                                            dof_handler);

    scratch_data->create_partitioning();

    simulation_case->set_time_boundary_conditions(
      simulation_case->parameters.time_stepping.start_time);

    // create the matrix-free object
    scratch_data->build(true, true);

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
  CompressibleFlowProblem<dim>::initialize()
  {
    // setup DoFHandler
    dof_handler.reinit(*simulation_case->triangulation);

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
      comp_flow_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.flow.fe));

      comp_flow_dof_idx = scratch_data->attach_dof_handler(dof_handler);

      scratch_data->attach_constraint_matrix(constraints);
    }

    setup_dof_system();

    // initialize the time iterator
    time_iterator =
      std::make_unique<TimeIterator<double>>(simulation_case->parameters.time_stepping);

    // initialize compressible flow operation
    comp_flow_operation = std::make_unique<CompressibleFlowOperation<dim, double>>(
      *scratch_data, simulation_case->parameters.flow, comp_flow_dof_idx, comp_flow_quad_idx);

    // set boundary conditions
    comp_flow_operation->set_inflow_boundary(
      simulation_case->get_boundary_condition("inflow", "compressible_flow"));

    comp_flow_operation->set_subsonic_outflow_with_fixed_static_pressure(
      simulation_case->get_boundary_condition("outflow_fixed_pressure", "compressible_flow"));

    comp_flow_operation->set_subsonic_outflow_with_fixed_energy(
      simulation_case->get_boundary_condition("outflow_fixed_energy", "compressible_flow"));

    comp_flow_operation->set_slip_wall_boundary(
      simulation_case->get_boundary_condition("slip_wall", "compressible_flow"));

    comp_flow_operation->set_no_slip_adiabatic_wall_boundary(
      simulation_case->get_boundary_condition("no_slip_wall", "compressible_flow"));

    // initialize operation
    comp_flow_operation->reinit();

    // set initial condition for the flow field
    comp_flow_operation->set_initial_condition(
      *simulation_case->get_initial_condition("compressible_flow"));

    // set body force
    if (dim > 1 && simulation_case->parameters.flow.gravity_constant > 0.)
      {
        std::unique_ptr<Functions::ConstantFunction<dim>> body_force =
          std::make_unique<Functions::ConstantFunction<dim>>(
            dim > 2 ?
              std::vector<double>({0., 0., -simulation_case->parameters.flow.gravity_constant}) :
              std::vector<double>({0., -simulation_case->parameters.flow.gravity_constant}));
        comp_flow_operation->set_body_force(std::move(body_force));
      }

    // initialize postprocessor
    post_processor =
      std::make_unique<Postprocessor<dim>>(scratch_data->get_mpi_comm(comp_flow_dof_idx),
                                           simulation_case->parameters.output,
                                           simulation_case->parameters.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(comp_flow_dof_idx),
                                           scratch_data->get_pcout(1));

    // initialize profiling
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<double>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim>
  void
  CompressibleFlowProblem<dim>::output_results(const unsigned int time_step,
                                               const double       current_time)
  {
    if (!post_processor->is_output_timestep(time_step, current_time) &&
        !simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      comp_flow_operation->attach_output_vectors(data_out);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         current_time,
                                         simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing &&
        post_processor->is_output_timestep(time_step, current_time))
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, current_time);
  }


  template class CompressibleFlowProblem<1>;
  template class CompressibleFlowProblem<2>;
  template class CompressibleFlowProblem<3>;

} // namespace MeltPoolDG::Flow


int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace MeltPoolDG;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;
  // check command line arguments
  if (argc == 1)
    {
      if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        std::cout << "ERROR: No .json parameter files has been provided!" << std::endl;
      return 1;
    }
  else if (argc == 2)
    {
      input_file = std::string(argv[argc - 1]);
      run_simulation<Flow::CompressibleFlowCaseParameters<double>,
                     Flow::CompressibleFlowCase,
                     Flow::CompressibleFlowProblem>(input_file, mpi_comm);
    }
  else if (argc == 3 &&
           ((std::string(argv[1]) == "--help") || (std::string(argv[1]) == "--help-detail")))
    {
      input_file = std::string(argv[argc - 1]);

      ParameterHandler                             prm;
      Flow::CompressibleFlowCaseParameters<double> parameters;
      parameters.process_parameters_file(prm, input_file);

      if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        parameters.print_parameters(prm,
                                    std::cout,
                                    std::string(argv[1]) == "--help-detail" /*print_details*/);

      return 0;
    }
  else
    AssertThrow(false, ExcMessage("no input file specified"));

  return 0;
}