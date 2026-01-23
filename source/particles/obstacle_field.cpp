
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/particle_handler.h>

#include "meltpooldg/particles/dem_util.hpp"
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <fstream>
#include <functional>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData<number>       &data,
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping)
  : data(data)
  , obstacle_data_structure(triangulation, mapping)
  , obstacle_handler_vector_views(
      std::vector<std::reference_wrapper<dealii::Particles::ParticleHandler<dim>>>(
        {obstacle_data_structure}),
      1)
  , mpi_communicator(triangulation.get_mpi_communicator())
{
  auto [obstacle_locations, obstacle_properties] = read_obstacle_state_input_file();
  insert_obstacles(triangulation, obstacle_locations, obstacle_properties);
  obstacle_data_structure.reinit();
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData<number>              &data,
  const dealii::Triangulation<dim>        &triangulation,
  const dealii::Mapping<dim>              &mapping,
  std::vector<dealii::Point<dim, number>> &obstacle_locations,
  std::vector<std::vector<number>>        &obstacle_properties)
  : data(data)
  , obstacle_data_structure(triangulation, mapping)
  , obstacle_handler_vector_views(
      std::vector<std::reference_wrapper<dealii::Particles::ParticleHandler<dim>>>(
        {obstacle_data_structure}),
      1)
  , mpi_communicator(triangulation.get_mpi_communicator())
{
  insert_obstacles(triangulation, obstacle_locations, obstacle_properties);
  obstacle_data_structure.reinit();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::advance_time(const number current_time,
                                                                   const number time_step)
{
  compute_loads_on_obstacles();

  symplectic_euler_advance_time_step<
    dim,
    double,
    ParticleHandlerBlockVectorView<dim, number, ParticleDataStructure>>(
    current_time,
    time_step,
    obstacle_handler_vector_views.location,
    obstacle_handler_vector_views.translational_velocity,
    obstacle_handler_vector_views.translational_acceleration,
    obstacle_handler_vector_views.angular_velocity,
    obstacle_handler_vector_views.angular_acceleration,
    [&](number, ParticleHandlerBlockVectorView<dim, number, ParticleDataStructure> &) {
      for (auto &obstacle : obstacle_data_structure.locally_owned_particle_range())
        {
          obstacle.set_linear_acceleration(obstacle.get_force() /
                                           obstacle.get_property(ObstacleType::Properties::mass));
        }
    },
    [&](number, ParticleHandlerBlockVectorView<dim, number, ParticleDataStructure> &) {
      for (auto &obstacle : obstacle_data_structure.locally_owned_particle_range())
        {
          obstacle.set_angular_acceleration(
            obstacle.get_torque() /
            obstacle.get_property(ObstacleType::Properties::moment_of_inertia));
        }
    },
    [&]() { obstacle_data_structure.update_ghost_particles(); });

  // Temporary workaround: prevent particles from moving below ground level. We check whether the
  // particle’s vertical coordinate (y in 2D, z in 3D) is less than its radius, i.e., meaning part
  // of the particle would lie below the ground. If so, we clamp the particle to ground contact by
  // setting its center height equal to its radius and zeroing its vertical velocity.
  for (auto &particle : obstacle_data_structure.locally_owned_particle_range())
    if (particle.get_location()[dim - 1] - particle.get_property(ObstacleType::Properties::radius) <
        0)
      {
        particle.get_location()[dim - 1] = particle.get_property(ObstacleType::Properties::radius);
        auto velocity                    = particle.template get_linear_velocity<number>();
        velocity[dim - 1]                = 0;
        particle.set_velocity(velocity);
      }
}


template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim> &dst,
  const dealii::CellAccessor<dim>      &cell) const
{
  return obstacle_data_structure.get_obstacles_in_cell(dst, cell);
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_obstacles_in_cell_batch(
  dealii::Particles::PropertyPool<dim>  &dst,
  const dealii::MatrixFree<dim, number> &matrix_free,
  const unsigned int                     cell_batch_id) const
{
  return obstacle_data_structure.get_obstacles_in_cell_batch(dst, matrix_free, cell_batch_id);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::prepare_for_serialization()
{
  obstacle_data_structure.prepare_for_serialization();
}

template <int dim, typename number, typename ObstacleType>
std::vector<MeltPoolDG::AMR::AMRRegion<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_refinement_regions() const
{
  if (not data.amr.do_refine_obstacles)
    return std::vector<AMR::AMRRegion<dim, number>>();

  std::vector<AMR::SphericalShellAMRRegion<dim, number>> local_ref_regions;
  local_ref_regions.reserve(obstacle_data_structure.n_locally_owned_particles());
  for (const auto &obstacle : obstacle_data_structure.locally_owned_particle_range())
    local_ref_regions.emplace_back(obstacle.get_location(),
                                   data.amr.inner_fractional_distance_to_surface *
                                     obstacle.get_property(ObstacleType::Properties::radius),
                                   data.amr.outer_fractional_distance_to_surface *
                                     obstacle.get_property(ObstacleType::Properties::radius));

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
  if (loads.empty())
    return;

  // Reset current particle forces and torques
  for (auto &obstacle : obstacle_data_structure.locally_owned_particle_range())
    {
      obstacle.set_force(dealii::Tensor<1, dim, number>());
      obstacle.set_torque(dealii::Tensor<1, axial_dim<dim>, number>());
    }

  // Update global particle property pool in case any of the loads needs an up-to-date version.
  obstacle_data_structure.broadcast_global_particles();

  // Accumulate all forces acting on the particles
  for (const auto &load_type : loads)
    load_type.add_load_to_obstacles(*this);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::print_accumulated_obstacle_force_norm(
  const dealii::ConditionalOStream pout) const
{
  dealii::Tensor<1, dim, number> accumulated_force;
  for (const auto &obstacle : obstacle_data_structure.locally_owned_particle_range())
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
  const dealii::Triangulation<dim>        &triangulation,
  std::vector<dealii::Point<dim, number>> &obstacle_locations,
  std::vector<std::vector<number>>        &obstacle_properties)
{
  obstacle_data_structure.insert_obstacles(triangulation, obstacle_locations, obstacle_properties);
}

template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::locally_owned_particle_range()
{
  return obstacle_data_structure.locally_owned_particle_range();
}


template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::global_particle_range()
{
  return obstacle_data_structure.global_particle_range();
}

template class MeltPoolDG::ObstacleField<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::ObstacleField<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::ObstacleField<3, double, MeltPoolDG::SphericalParticle<3, double>>;
