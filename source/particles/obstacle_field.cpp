
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/particle_handler.h>

#include "meltpooldg/particles/dem_time_integrators.hpp"
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <fstream>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData<number>       &data,
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping,
  dealii::TimerOutput              &timer,
  const ObstacleDataStructureType   obstacle_data_structure_type)
  : data(data)
  , obstacle_data_structure(triangulation, mapping, timer)
  , mpi_communicator(triangulation.get_mpi_communicator())
  , timer(timer)
{
  auto [obstacle_locations, obstacle_properties] = read_obstacle_state_input_file();
  insert_obstacles(obstacle_locations, obstacle_properties);
  obstacle_data_structure.reinit();
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData<number>              &data,
  const dealii::Triangulation<dim>        &triangulation,
  const dealii::Mapping<dim>              &mapping,
  std::vector<dealii::Point<dim, number>> &obstacle_locations,
  std::vector<std::vector<number>>        &obstacle_properties,
  dealii::TimerOutput                     &timer,
  const ObstacleDataStructureType          obstacle_data_structure_type)
  : data(data)
  , obstacle_data_structure(triangulation, mapping, timer)
  , mpi_communicator(triangulation.get_mpi_communicator())
  , timer(timer)
{
  insert_obstacles(obstacle_locations, obstacle_properties);
  obstacle_data_structure.reinit();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::advance_time(const number,
                                                                   const number time_step)
{
  dealii::TimerOutput::Scope t(timer, "advance obstacle field in time");
  compute_loads_on_obstacles();

  symplectic_euler_advance_time_step<dim, number, ObstacleType>(timer,
                                                                time_step,
                                                                locally_owned_particle_range());

  obstacle_data_structure.sort_particles_into_subdomains_and_cells();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::prepare_for_serialization()
{
  obstacle_data_structure.prepare_for_serialization();
}

template <int dim, typename number, typename ObstacleType>
std::vector<MeltPoolDG::AMR::AMRRegion<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_refinement_regions()
{
  if (not data.amr.do_refine_obstacles)
    return std::vector<AMR::AMRRegion<dim, number>>();

  std::vector<AMR::SphericalShellAMRRegion<dim, number>> local_ref_regions;
  local_ref_regions.reserve(obstacle_data_structure.n_locally_owned_particles());
  for (const DEMParticleAccessor<dim, number> &obstacle :
       obstacle_data_structure.locally_owned_particle_range())
    local_ref_regions.emplace_back(obstacle.get_location(),
                                   data.amr.inner_fractional_distance_to_surface *
                                     obstacle.radius(),
                                   data.amr.outer_fractional_distance_to_surface *
                                     obstacle.radius());

  std::vector<std::vector<AMR::SphericalShellAMRRegion<dim, number>>> global_ref_regions =
    dealii::Utilities::MPI::all_gather(mpi_communicator, local_ref_regions);

  std::vector<AMR::AMRRegion<dim, number>> final_amr_regions;
  final_amr_regions.reserve(obstacle_data_structure.n_global_particles());
  for (unsigned i = 0; i < global_ref_regions.size(); ++i)
    for (unsigned j = 0; j < global_ref_regions[i].size(); ++j)
      final_amr_regions.emplace_back(std::move(global_ref_regions[i][j]));

  return final_amr_regions;
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::compute_loads_on_obstacles()
{
  dealii::TimerOutput::Scope t(timer, "compute obstacle loads");
  if (loads.empty())
    return;

  // Reset current particle forces and torques
  for (DEMParticleAccessor<dim, number> &obstacle :
       obstacle_data_structure.locally_owned_particle_range())
    {
      obstacle.set_force(dealii::Tensor<1, dim, number>());
      obstacle.set_torque(dealii::Tensor<1, axial_dim<dim>, number>());
    }

  // Update ghost particle properties to ensure that all particles have the most up-to-date
  // information available for load computation. This is especially important if any of the loads
  // depend on interactions between particles, such as contact or cohesive forces.
  obstacle_data_structure.update_ghost_particle_properties();

  // Accumulate all forces acting on the particles
  for (const auto &load_type : loads)
    load_type.add_load_to_obstacles(*this);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::print_accumulated_obstacle_force_norm(
  const dealii::ConditionalOStream pout)
{
  dealii::Tensor<1, dim, number> accumulated_force;
  for (DEMParticleAccessor<dim, number> &obstacle :
       obstacle_data_structure.locally_owned_particle_range())
    accumulated_force += obstacle.get_force();

  accumulated_force = dealii::Utilities::MPI::sum(accumulated_force, mpi_communicator);

  std::ostringstream output;
  output << std::scientific << std::setprecision(4)
         << "accumulated norm: " << accumulated_force.norm();
  Journal::print_line(pout, output.str(), "obstacle_force");
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::deserialize()
{
  // Assumes that triangulation.load() has already been called!
  obstacle_data_structure.deserialize();
}

template <int dim, typename number, typename ObstacleType>
std::pair<std::vector<dealii::Point<dim, number>>, std::vector<std::vector<number>>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::read_obstacle_state_input_file()
{
  std::vector<dealii::Point<dim>> particle_locations;
  // Make global particle properties vector
  std::vector<std::vector<number>> properties{};
  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::fstream file;
      file.open(data.obstacle_state_input_file, std::ios::in);
      AssertThrow(!(file.fail()),
                  dealii::ExcMessage("Unable to open particle data file \"" +
                                     data.obstacle_state_input_file + "\". Aborting!"));
      std::string line;
      std::getline(file, line); // Ignore the first line
      while (std::getline(file, line))
        {
          auto obstacle_properties = ObstacleType::read_state_input(line);
          properties.push_back(obstacle_properties.first);
          particle_locations.push_back(obstacle_properties.second);
        }
    }

  return {particle_locations, properties};
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::insert_obstacles(
  std::vector<dealii::Point<dim, number>> &obstacle_locations,
  std::vector<std::vector<number>>        &obstacle_properties)
{
  obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
}

template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::locally_owned_particle_range()
{
  return obstacle_data_structure.locally_owned_particle_range();
}

template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ghost_particle_range()
{
  return obstacle_data_structure.ghost_particle_range();
}

template class MeltPoolDG::ObstacleField<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::ObstacleField<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::ObstacleField<3, double, MeltPoolDG::SphericalParticle<3, double>>;
