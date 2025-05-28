#include "cfd_dem_application.hpp"

#include <deal.II/base/mpi.h>

#include "meltpooldg/utilities/cell_monitor.hpp"
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operation.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_fluid_force.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include "cfd_dem_case.hpp"

namespace MeltPoolDG
{
  template <int dim, typename number>
  void
  CfdDemApplication<dim, number>::run()
  {
    initialize();

    while (!time_iterator->is_finished())
      {
        // use CFL condition to compute time step size if required
        if (simulation_case->parameters.flow.do_cfl_time_stepping)
          time_iterator->set_current_time_increment(comp_flow_operation.compute_time_step_size(),
                                                    std::numeric_limits<number>::max());

        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));

        comp_flow_operation.solve(time_iterator->get_current_time(),
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

  template <int dim, typename number>
  void
  CfdDemApplication<dim, number>::setup_dof_system()
  {
    // distribute DoFs
    comp_flow_operation.distribute_dofs(dof_handler);

    scratch_data->create_partitioning();

    simulation_case->set_time_boundary_conditions(
      simulation_case->parameters.time_stepping.start_time);

    // create the matrix-free object
    scratch_data->build(true, true, false, false);

    // print mesh information
    {
      const ScopedName sc("compFlow::cells");
      CellMonitor<number>::add_info(sc,
                                    scratch_data->get_triangulation().n_global_active_cells(),
                                    scratch_data->get_min_cell_size(),
                                    scratch_data->get_max_cell_size());
    }
  }

  template <int dim, typename number>
  void
  CfdDemApplication<dim, number>::initialize()
  {
    // setup DoFHandler
    dof_handler.reinit(*simulation_case->triangulation);

    // setup scratch data
    {
      scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
        simulation_case->mpi_communicator, simulation_case->parameters.base.verbosity_level, true);
      // set up mapping
      scratch_data->set_mapping(
        FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.flow.fe));

      // create quadrature rule
      comp_flow_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.flow.fe));

      comp_flow_dof_idx = scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_constraint_matrix(constraints);
    }

    // initialize the time iterator
    time_iterator = std::make_shared<TimeIntegration::TimeIterator<number>>(
      simulation_case->parameters.time_stepping);

    // initilaize obstacle field
    obstacle_field = std::make_unique<ObstacleField<dim, number, SphericalParticle<dim, number>>>(
      simulation_case->parameters.obstacle_data,
      scratch_data->get_triangulation(),
      scratch_data->get_mapping());

    // initialize external fluid forces
    std::unique_ptr<
      BrinkmanPenalizationFluidForceRightHandSideContribution<dim,
                                                              number,
                                                              SphericalParticle<dim, number>>>
      external_fluid_force = std::make_unique<
        BrinkmanPenalizationFluidForceRightHandSideContribution<dim,
                                                                number,
                                                                SphericalParticle<dim, number>>>(
        *obstacle_field, simulation_case->parameters.brinkman_penalization_data);

    // initialize compressible flow operation
    std::unique_ptr<Flow::DGCompressibleFlowOperation<dim, number>> operation =
      std::make_unique<Flow::DGCompressibleFlowOperation<dim, number>>(
        *scratch_data,
        simulation_case->parameters.flow,
        comp_flow_dof_idx,
        comp_flow_quad_idx,
        std::move(external_fluid_force));

    comp_flow_operation = Flow::CompressibleFlowOperation<dim, number>(std::move(operation));

    setup_dof_system();

    // set boundary conditions
    comp_flow_operation.set_boundary_conditions(simulation_case, "cfd_dem");

    // initialize operation
    comp_flow_operation.reinit();

    // set initial condition for the flow field
    comp_flow_operation.set_initial_condition(*simulation_case->get_initial_condition("cfd_dem"));

    // set fluid body force
    if (dim > 1 and simulation_case->parameters.flow.gravity_constant > 0.)
      {
        std::unique_ptr<dealii::Functions::ConstantFunction<dim>> body_force =
          std::make_unique<dealii::Functions::ConstantFunction<dim>>(
            dim > 2 ?
              std::vector<number>({0., 0., -simulation_case->parameters.flow.gravity_constant}) :
              std::vector<number>({0., -simulation_case->parameters.flow.gravity_constant}));
        comp_flow_operation.set_body_force(std::move(body_force));
      }

    // initialize postprocessor
    post_processor =
      std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(comp_flow_dof_idx),
                                                   simulation_case->parameters.output,
                                                   simulation_case->parameters.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(
                                                     comp_flow_dof_idx),
                                                   scratch_data->get_pcout(2));

    // initialize profiling
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim, typename number>
  void
  CfdDemApplication<dim, number>::output_results(const unsigned int time_step,
                                                 const number       current_time)
  {
    if (not post_processor->is_output_timestep(time_step, current_time) and
        not simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &data_out) {
      comp_flow_operation.attach_output_vectors(data_out);
    };

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(),
      current_time,
      simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing and
        post_processor->is_output_timestep(time_step, current_time))
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, current_time);
  }

  template class CfdDemApplication<2, double>;
  template class CfdDemApplication<3, double>;

} // namespace MeltPoolDG
int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::CfdDemCaseParameters<double>,
                           MeltPoolDG::CfdDemCase,
                           MeltPoolDG::CfdDemApplication>(argc, argv, mpi_comm);
  return 0;
}
