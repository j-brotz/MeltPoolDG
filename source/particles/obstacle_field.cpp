
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_tools.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <functional>
#include <vector>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData<number>       &data,
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping)
  : data(data)
  , obstacle_handler(triangulation, mapping, ObstacleType::n_obstacle_properties)
  , obstacle_data_structure(obstacle_handler)
  , mpi_communicator(triangulation.get_mpi_communicator())
{
  auto [obstacle_locations, obstacle_properties] =
    read_particle_state_input_file<dim, number>(data.obstacle_state_input_file, mpi_communicator);
  insert_obstacles(triangulation, obstacle_locations, obstacle_properties);
  obstacle_data_structure.reinit();
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData<number>                    &data,
  const dealii::Triangulation<dim>              &triangulation,
  const dealii::Mapping<dim>                    &mapping,
  const std::vector<dealii::Point<dim, number>> &obstacle_locations,
  const std::vector<std::vector<number>>        &obstacle_properties)
  : data(data)
  , obstacle_handler(triangulation, mapping, ObstacleType::n_obstacle_properties)
  , obstacle_data_structure(obstacle_handler)
  , mpi_communicator(triangulation.get_mpi_communicator())
{
  insert_obstacles(triangulation, obstacle_locations, obstacle_properties);
  obstacle_data_structure.reinit();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::advance_time(const number time_step)
{
  compute_loads_on_obstacles();

  symplectic_euler_advance_time_step<dim, number, ObstacleType>(time_step,
                                                                locally_owned_particle_range());

  // Temporary workaround: prevent particles from moving below ground level. We check whether the
  // particle’s vertical coordinate (y in 2D, z in 3D) is less than its radius, i.e., meaning part
  // of the particle would lie below the ground. If so, we clamp the particle to ground contact by
  // setting its center height equal to its radius and zeroing its vertical velocity.
  for (auto &particle : obstacle_handler)
    if (particle.get_location()[dim - 1] -
          ObstacleType::get_property(particle, ObstacleType::Properties::radius) <
        0)
      {
        particle.get_location()[dim - 1] =
          ObstacleType::get_property(particle, ObstacleType::Properties::radius);
        auto velocity     = ObstacleType::template get_velocity<number>(particle);
        velocity[dim - 1] = 0;
        ObstacleType::set_velocity(particle, velocity);
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
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim>                               &dst,
  const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const
{
  return obstacle_data_structure.get_obstacles_in_cell(dst, cells);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::prepare_for_serialization()
{
  obstacle_handler.prepare_for_serialization();
}

template <int dim, typename number, typename ObstacleType>
std::vector<MeltPoolDG::AMR::AMRRegion<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_refinement_regions() const
{
  if (not data.amr.do_refine_obstacles)
    return std::vector<AMR::AMRRegion<dim, number>>();

  std::vector<AMR::SphericalShellAMRRegion<dim, number>> local_ref_regions;
  local_ref_regions.reserve(obstacle_handler.n_locally_owned_particles());
  for (const auto &obstacle : obstacle_handler)
    local_ref_regions.emplace_back(obstacle.get_location(),
                                   data.amr.inner_fractional_distance_to_surface *
                                     ObstacleType::get_characteristic_length(obstacle),
                                   data.amr.outer_fractional_distance_to_surface *
                                     ObstacleType::get_characteristic_length(obstacle));

  std::vector<std::vector<AMR::SphericalShellAMRRegion<dim, number>>> global_ref_regions =
    dealii::Utilities::MPI::all_gather(mpi_communicator, local_ref_regions);

  std::vector<AMR::AMRRegion<dim, number>> final_amr_regions;
  final_amr_regions.reserve(obstacle_handler.n_global_particles());
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
  for (dealii::Particles::ParticleAccessor<dim> obstacle : obstacle_handler)
    {
      ObstacleType::set_force(dealii::Tensor<1, dim, number>(), obstacle);
      ObstacleType::set_torque(dealii::Tensor<1, ObstacleType::size_angular_velocity, number>(),
                               obstacle);
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
  for (dealii::Particles::ParticleAccessor<dim> obstacle : obstacle_handler)
    accumulated_force += ObstacleType::get_force(obstacle);

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
  obstacle_handler.deserialize();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::insert_obstacles(
  const dealii::Triangulation<dim>              &triangulation,
  const std::vector<dealii::Point<dim, number>> &obstacle_locations,
  const std::vector<std::vector<number>>        &obstacle_properties)
{
  Assert(obstacle_locations.size() == obstacle_properties.size(),
         dealii::ExcMessage(
           "The number of particle locations must match the number of particle properties."));

  std::vector<dealii::BoundingBox<dim>> local_bounding_box =
    dealii::GridTools::compute_mesh_predicate_bounding_box(
      triangulation, dealii::IteratorFilters::LocallyOwnedCell());
  std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_box =
    dealii::Utilities::MPI::all_gather(mpi_communicator, local_bounding_box);

  obstacle_handler.insert_global_particles(
    dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ?
      obstacle_locations :
      std::vector<dealii::Point<dim, number>>{},
    global_bounding_box,
    dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ?
      obstacle_properties :
      std::vector<std::vector<number>>{});

  obstacle_data_structure.reinit();
}

template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::locally_owned_particle_range()
{
  return std::ranges::subrange<ParticleIterator<dim, number>>(
    ParticleIterator<dim, number>(obstacle_handler.begin()),
    ParticleIterator<dim, number>(obstacle_handler.end()));
}


template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::global_particle_range()
{
  return std::ranges::subrange<ParticleIterator<dim, number>>(
    ParticleIterator<dim, number>(obstacle_data_structure.get_global_particle_properties(), 0),
    ParticleIterator<dim, number>(
      obstacle_data_structure.get_global_particle_properties(),
      obstacle_data_structure.get_global_particle_properties().n_registered_slots()));
}

template class MeltPoolDG::ObstacleField<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::ObstacleField<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::ObstacleField<3, double, MeltPoolDG::SphericalParticle<3, double>>;
