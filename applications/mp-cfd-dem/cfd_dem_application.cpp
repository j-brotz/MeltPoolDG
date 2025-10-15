#include "cfd_dem_application.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include "meltpooldg/particles/obstacle_forces.hpp"
#include "meltpooldg/utilities/cell_monitor.hpp"
#include "meltpooldg/utilities/journal.hpp"
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operation.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_fluid_to_obstacle.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_obstacle_to_fluid.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/amr_indicators.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <memory>

#include "cfd_dem_case.hpp"

namespace MeltPoolDG
{
  /**
   * Internal helper function to print # cells and # dofs to the console.
   *
   * @param pcout Output stream used for console output.
   * @param tria_name Name of the triangulation printed at the beginning of line.
   * @param tria Triangulation from which # cells is taken from.
   * @param dof_handler Dof handler from which # dofs is taken from.
   */
  template <int dim>
  void
  print_triangulation_info(const ConditionalOStream         &pcout,
                           const std::string                &tria_name,
                           const dealii::Triangulation<dim> &tria,
                           const dealii::DoFHandler<dim>    &dof_handler)
  {
    std::ostringstream str;
    str << tria_name << ": " << std::setw(12) << std::right
        << "# cells: " << tria.n_global_active_cells() << ", "
        << "# dofs: " << dof_handler.n_dofs();
    Journal::print_line(pcout, str.str(), "");
  }

  template <int dim, typename number>
  void
  CfdDemApplication<dim, number>::run()
  {
    initialize();

    Journal::print_start(scratch_data->get_pcout(1));

    print_triangulation_info(scratch_data->get_pcout(1),
                             "Initial flow grid",
                             *this->simulation_case->triangulation,
                             scratch_data->get_dof_handler(comp_flow_dof_idx));
    Journal::print_decoration_line(scratch_data->get_pcout(1));

    // output the initial condition
    output_results(time_iterator->get_current_time_step_number(),
                   time_iterator->get_current_time());

    while (not time_iterator->is_finished())
      {
        // use CFL condition to compute time step size if required
        if (simulation_case->parameters.flow.do_cfl_time_stepping)
          time_iterator->set_current_time_increment(comp_flow_operation->compute_time_step_size(),
                                                    std::numeric_limits<number>::max());

        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));

        comp_flow_operation->solve(time_iterator->get_current_time(),
                                   time_iterator->get_current_time_increment());

        if (this->simulation_case->parameters.obstacle_data.stationary_obstacles)
          obstacle_field->compute_loads_on_obstacles();
        else
          {
            // The advance time function internally calls compute_loads_on_obstacles(). Therefore
            // this is not done explicitly here.
            obstacle_field->advance_time(time_iterator->get_current_time(),
                                         time_iterator->get_current_time_increment());
          }

        // do output if requested
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

        if (profiling_monitor and profiling_monitor->now())
          {
            profiling_monitor->print(scratch_data->get_pcout(1),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }

        refine_mesh();
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
    comp_flow_operation->distribute_dofs(dof_handler);

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
  CfdDemApplication<dim, number>::setup_amr_indicator()
  {
    if (not this->simulation_case->parameters.amr.do_amr)
      return;

    if (scratch_data->get_degree(comp_flow_dof_idx) < 2)
      amr_indicator.add_indicator(
        std::make_unique<AMR::JumpIndicator<dim, number>>(scratch_data->get_matrix_free(),
                                                          comp_flow_operation->get_solution(),
                                                          comp_flow_dof_idx,
                                                          comp_flow_quad_idx,
                                                          0));
    else
      {
        amr_indicator.add_indicator(
          std::make_unique<AMR::JumpIndicator<dim, number>>(scratch_data->get_matrix_free(),
                                                            comp_flow_operation->get_solution(),
                                                            comp_flow_dof_idx,
                                                            comp_flow_quad_idx,
                                                            0),
          1. / scratch_data->get_degree(0));
        amr_indicator.add_indicator(std::make_unique<AMR::SSEDIndicator<dim, number>>(
                                      scratch_data->get_matrix_free(),
                                      comp_flow_operation->get_solution(),
                                      comp_flow_dof_idx,
                                      comp_flow_quad_idx,
                                      0,
                                      scratch_data->get_degree(comp_flow_dof_idx)),
                                    1);
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

    // initialize obstacle field
    obstacle_field = std::make_unique<ObstacleField<dim, number, SphericalParticle<dim, number>>>(
      simulation_case->parameters.obstacle_data,
      scratch_data->get_triangulation(),
      scratch_data->get_mapping());

    // initialize compressible flow operation
    comp_flow_operation = std::make_unique<Flow::DGCompressibleFlowOperation<dim, number>>(
      *scratch_data,
      simulation_case->parameters.flow,
      simulation_case->parameters.material,
      comp_flow_dof_idx,
      comp_flow_quad_idx);

    setup_dof_system();

    // set boundary conditions
    comp_flow_operation->set_boundary_conditions(simulation_case, "cfd_dem");

    // initialize operation
    comp_flow_operation->reinit();

    // set initial condition for the flow field
    comp_flow_operation->set_initial_condition(*simulation_case->get_initial_condition("cfd_dem"));

    // initialize volume penalization fluid forces
    auto external_fluid_force_residual = std::make_shared<
      BrinkmanPenalizationResidualContribution<dim, number, SphericalParticle<dim, number>>>(
      *obstacle_field, simulation_case->parameters.brinkman_penalization_data);

    auto external_fluid_force_jacobian = std::make_shared<
      BrinkmanPenalizationJacobianContribution<dim, number, SphericalParticle<dim, number>>>(
      *obstacle_field, simulation_case->parameters.brinkman_penalization_data);

    comp_flow_operation->add_external_force(external_fluid_force_residual,
                                            external_fluid_force_jacobian);

    // set fluid body force
    if (dim > 1 and simulation_case->parameters.flow.gravity_constant > 0.)
      {
        std::unique_ptr<dealii::Functions::ConstantFunction<dim>> body_force =
          std::make_unique<dealii::Functions::ConstantFunction<dim>>(
            dim > 2 ?
              std::vector<number>({0., 0., -simulation_case->parameters.flow.gravity_constant}) :
              std::vector<number>({0., -simulation_case->parameters.flow.gravity_constant}));
        comp_flow_operation->set_body_force(std::move(body_force));
      }

    // setup amr indicator
    setup_amr_indicator();

    // add relevant obstacle forces to obstacle field
    obstacle_field->add_load_type(
      BrinkmanObstacleForce<dim, number, SphericalParticle<dim, number>>(
        comp_flow_operation->get_solution(),
        scratch_data->get_matrix_free(),
        comp_flow_dof_idx,
        comp_flow_quad_idx,
        BrinkmanPenalizationCellScratchData<dim, number, SphericalParticle<dim, number>>(
          *obstacle_field, simulation_case->parameters.brinkman_penalization_data)));

    obstacle_field->add_load_type(
      ObstacleGravitationalForce<dim, number, SphericalParticle<dim, number>>(
        this->simulation_case->parameters.flow.gravity_constant));

    // initialize postprocessor
    post_processor =
      std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(comp_flow_dof_idx),
                                                   simulation_case->parameters.output,
                                                   simulation_case->parameters.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(
                                                     comp_flow_dof_idx),
                                                   scratch_data->get_pcout(2));

    const auto [property_names, property_component_interpretations] =
      SphericalParticle<dim, number>::get_property_names_and_component_interpretation();

    post_processor->register_obstacle_output(&obstacle_field->get_particle_handler(),
                                             property_names,
                                             property_component_interpretations);

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
      comp_flow_operation->attach_output_vectors(data_out);
    };

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(),
      current_time,
      simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing and
        post_processor->is_output_timestep(time_step, current_time))
      {
        simulation_case->do_postprocessing(generic_data_out);
        obstacle_field->print_accumulated_obstacle_force_norm(scratch_data->get_pcout(1));
      }

    // postprocessing
    post_processor->process(time_step, generic_data_out, current_time);
  }

  template <int dim, typename number>
  void
  CfdDemApplication<dim, number>::refine_mesh()
  {
    if (not simulation_case->parameters.amr.do_amr)
      return;

    Assert(
      not amr_indicator.empty(),
      dealii::ExcMessage(
        "No AMR indicator has been set up. Without indicator adaptive mesh refinement is not possible."));

    const AMR::MarkCellsForRefinementType<dim> mark_cells_for_refinement =
      [&](dealii::Triangulation<dim> &tria) {
        return AMR::mark_fixed_fraction<dim, number>(tria,
                                                     amr_indicator.compute_indicator(tria),
                                                     this->simulation_case->parameters.amr);
      };

    const std::function<void(std::vector<VectorType *> &)> attach_vectors =
      [&](std::vector<VectorType *> &in) { in.push_back(&comp_flow_operation->get_solution()); };

    std::function<void()> pre = [&] {
      obstacle_field->get_particle_handler().prepare_for_coarsening_and_refinement();

      // print mesh refinement info to console
      Journal::print_decoration_line(scratch_data->get_pcout(1));
      std::ostringstream str_header;
      str_header << "Performing mesh refinement...";
      Journal::print_line(scratch_data->get_pcout(1), str_header.str(), "");
      print_triangulation_info(scratch_data->get_pcout(1),
                               "Flow grid before refinement",
                               *this->simulation_case->triangulation,
                               scratch_data->get_dof_handler(comp_flow_dof_idx));
    };

    std::function<void()> post = [&] {
      obstacle_field->get_particle_handler().unpack_after_coarsening_and_refinement();
      print_triangulation_info(scratch_data->get_pcout(1),
                               "Flow grid after refinement ",
                               *this->simulation_case->triangulation,
                               scratch_data->get_dof_handler(comp_flow_dof_idx));
      Journal::print_decoration_line(scratch_data->get_pcout(1));
    };

    std::function<void()> setup_dofs = [&] {
      comp_flow_operation->distribute_dofs(dof_handler);
      scratch_data->create_partitioning();
      scratch_data->build(true, true, false, false);
      comp_flow_operation->reinit();
    };

    AMR::refine_grid(mark_cells_for_refinement,
                     attach_vectors,
                     setup_dofs,
                     this->simulation_case->parameters.amr,
                     scratch_data->get_dof_handler(comp_flow_dof_idx),
                     time_iterator->get_current_time_step_number(),
                     post,
                     pre);
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
