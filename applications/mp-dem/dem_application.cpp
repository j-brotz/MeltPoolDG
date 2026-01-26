#include "dem_application.hpp"

#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operation.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_factory.hpp>
#include <meltpooldg/fluid_structure_interaction/stokes_law.hpp>
#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/amr_indicators.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/restart.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <functional>
#include <memory>

#include "dem_case.hpp"

namespace MeltPoolDG
{

  template <int dim, typename number>
  void
  DemApplication<dim, number>::run()
  {
    initialize();

    if (restart_monitor->do_load())
      load_state_from_restart_file();

    Journal::print_start(scratch_data->get_pcout(1));

    Journal::print_decoration_line(scratch_data->get_pcout(1));

    // output the initial condition
    output_results(time_iterator->get_current_time_step_number(),
                   time_iterator->get_current_time());

    while (not time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));

        obstacle_field->advance_time(time_iterator->get_current_time(),
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

        if (restart_monitor->do_save())
          save_state_to_restart_file();
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
  DemApplication<dim, number>::initialize()
  {
    // setup scratch data
    {
      scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
        simulation_case->mpi_communicator, simulation_case->parameters.base.verbosity_level, true);
      // set up mapping
      scratch_data->set_mapping(
        FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.base.fe));
    }

    // initialize the time iterator
    time_iterator = std::make_shared<TimeIntegration::TimeIterator<number>>(
      simulation_case->parameters.time_stepping);

    // initialize obstacle field
    obstacle_field = std::make_unique<ObstacleField<dim, number, ObstacleType>>(
      simulation_case->parameters.obstacle_data,
      *simulation_case->triangulation,
      scratch_data->get_mapping());

    obstacle_field->add_load_type(ObstacleGravitationalForce<dim, number, ObstacleType>(
      this->simulation_case->parameters.obstacle_data.gravity_constant));

    SphericalParticleContactForce<dim, number, ObstacleType> contact_force(
      simulation_case->parameters.obstacle_contact, *time_iterator);

    dealii::Tensor<1, dim, number> ground_normal;
    ground_normal[dim - 1] = 1;
    contact_force.add_wall(
      std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(dealii::Point<dim, number>(),
                                                                      ground_normal));

    obstacle_field->add_load_type(contact_force);

    // initialize postprocessor
    post_processor = std::make_unique<Postprocessor<dim, number>>(
      simulation_case->triangulation->get_mpi_communicator(),
      simulation_case->parameters.output,
      simulation_case->parameters.time_stepping,
      scratch_data->get_mapping(),
      *simulation_case->triangulation,
      scratch_data->get_pcout(2));

    const auto [property_names, property_component_interpretations] =
      ObstacleType::get_property_names_and_component_interpretation();

    post_processor->register_obstacle_output(&obstacle_field->get_particle_handler(),
                                             property_names,
                                             property_component_interpretations);

    // initialize restart monitor
    restart_monitor =
      std::make_unique<Restart::RestartMonitor<number>>(this->simulation_case->parameters.restart,
                                                        *time_iterator);

    // initialize profiling
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim, typename number>
  void
  DemApplication<dim, number>::output_results(const unsigned int time_step,
                                              const number       current_time)
  {
    if (not post_processor->is_output_timestep(time_step, current_time) and
        not simulation_case->parameters.output.do_user_defined_postprocessing and
        not time_iterator->is_finished())
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &) {};

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
  DemApplication<dim, number>::load_state_from_restart_file()
  {
    Journal::print_line(scratch_data->get_pcout(1), "Loading state from restart file.");

    const std::string load_prefix = this->simulation_case->parameters.restart.prefix + "_" +
                                    std::to_string(this->simulation_case->parameters.restart.load);

    {
      std::ifstream                 ifs(load_prefix + "_problem.restart");
      boost::archive::text_iarchive ia(ifs);
      ia >> *this;
    }

    const std::function<void(DoFHandlerAndVectorDataType<dim, VectorType> &)> attach_vectors =
      [&](DoFHandlerAndVectorDataType<dim, VectorType> &) {};

    std::function<void()> setup_dof_system = [&]() {};

    std::function<void()> post = [&]() { obstacle_field->deserialize(); };

    Restart::deserialize_internal(attach_vectors, post, setup_dof_system, load_prefix);
  }

  template <int dim, typename number>
  void
  DemApplication<dim, number>::save_state_to_restart_file()
  {
    Journal::print_line(scratch_data->get_pcout(1), "Saving state to restart file.");

    restart_monitor->prepare_save();
    obstacle_field->prepare_for_serialization();

    std::ofstream ofs(simulation_case->parameters.restart.prefix + "_0_problem.restart");
    {
      boost::archive::text_oarchive oa(ofs);
      oa << *this;
    }

    const std::function<void(DoFHandlerAndVectorDataType<dim, VectorType> &)> attach_vectors =
      [&](DoFHandlerAndVectorDataType<dim, VectorType> &) {};

    const std::string save_prefix = this->simulation_case->parameters.restart.prefix + "_0";

    Restart::serialize_internal(attach_vectors, save_prefix);
  }

  template class DemApplication<2, double>;
  template class DemApplication<3, double>;
} // namespace MeltPoolDG

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::DemCaseParameters<double>,
                           MeltPoolDG::DemCase,
                           MeltPoolDG::DemApplication>(argc, argv, mpi_comm);
  return 0;
}
