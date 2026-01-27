
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/particle_handler.h>

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
  , obstacle_handler(triangulation, mapping, ObstacleType::n_obstacle_properties)
  , obstacle_handler_vector_views(
      std::vector<std::reference_wrapper<dealii::Particles::ParticleHandler<dim>>>(
        {obstacle_handler}),
      1)
  , obstacle_data_structure(obstacle_handler)
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
  , obstacle_handler(triangulation, mapping, ObstacleType::n_obstacle_properties)
  , obstacle_handler_vector_views(
      std::vector<std::reference_wrapper<dealii::Particles::ParticleHandler<dim>>>(
        {obstacle_handler}),
      1)
  , obstacle_data_structure(obstacle_handler)
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

  symplectic_euler_advance_time_step<dim, number, ParticleHandlerBlockVectorView<dim, number>>(
    current_time,
    time_step,
    obstacle_handler_vector_views.location,
    obstacle_handler_vector_views.translational_velocity,
    obstacle_handler_vector_views.translational_acceleration,
    obstacle_handler_vector_views.angular_velocity,
    obstacle_handler_vector_views.angular_acceleration,
    [&](number, ParticleHandlerBlockVectorView<dim, number> &) {
      for (auto &obstacle : obstacle_handler)
        {
          ObstacleType::set_acceleration(
            obstacle,
            ObstacleType::get_force(obstacle) /
              ObstacleType::get_property(obstacle, ObstacleType::Properties::mass));
        }
    },
    [&](number, ParticleHandlerBlockVectorView<dim, number> &) {
      for (auto &obstacle : obstacle_handler)
        {
          ObstacleType::set_angular_acceleration(
            obstacle,
            ObstacleType::get_torque(obstacle) /
              ObstacleType::get_property(obstacle, ObstacleType::Properties::moment_of_inertia));
        }
    },
    [&]() {
      obstacle_handler.exchange_ghost_particles(true);
      obstacle_handler.update_ghost_particles();
    });
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
      ObstacleType::set_torque(dealii::Tensor<1, axial_dim<dim>, number>(), obstacle);
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
