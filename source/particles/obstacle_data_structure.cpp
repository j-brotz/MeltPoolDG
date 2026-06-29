#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::ObstacleCompleteDomainSearch(
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping)
  : obstacle_handler(triangulation, mapping, ObstacleType::n_obstacle_properties)
  , properties_global_obstacles(ObstacleType::n_obstacle_properties)
{}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::ObstacleCompleteDomainSearch::
  ~ObstacleCompleteDomainSearch()
{
  deregister_property_pool();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::reinit()
{
  deregister_property_pool();
  properties_global_obstacles.clear();
  broadcast_global_particles();
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim> &dst,
  const dealii::CellAccessor<dim>      &cell) const
{
  std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> handles;
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles.n_registered_slots();
       ++src_handle)
    {
      if (ObstacleType::is_in_cell(properties_global_obstacles, src_handle, cell))
        {
          auto dst_handle = dst.register_particle();
          handles.emplace_back(dst_handle);
          dst.set_location(dst_handle, properties_global_obstacles.get_location(src_handle));
          auto dst_properties = dst.get_properties(dst_handle);
          auto src_properties = properties_global_obstacles.get_properties(src_handle);

          for (unsigned int n_property = 0; n_property < ObstacleType::n_obstacle_properties;
               ++n_property)
            {
              dst_properties[n_property] = src_properties[n_property];
            }
        }
    }
  return handles;
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim>                               &dst,
  const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const
{
  std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> handles;
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles.n_registered_slots();
       ++src_handle)
    {
      for (const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell : cells)
        {
          if (ObstacleType::is_in_cell(properties_global_obstacles, src_handle, *cell))
            {
              auto dst_handle = dst.register_particle();
              handles.emplace_back(dst_handle);
              dst.set_location(dst_handle, properties_global_obstacles.get_location(src_handle));
              auto dst_properties = dst.get_properties(dst_handle);
              auto src_properties = properties_global_obstacles.get_properties(src_handle);

              for (unsigned int n_property = 0; n_property < ObstacleType::n_obstacle_properties;
                   ++n_property)
                {
                  dst_properties[n_property] = src_properties[n_property];
                }
              break;
            }
        }
    }
  return handles;
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::broadcast_global_particles()
  const
{
  deregister_property_pool();
  properties_global_obstacles.clear();
  using Handle = typename dealii::Particles::PropertyPool<dim>::Handle;
  for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
       ++rank)
    {
      unsigned int rank_local_obstacles = obstacle_handler.n_locally_owned_particles();
      dealii::Utilities::MPI::broadcast(&rank_local_obstacles, 1, rank, mpi_communicator);
      for (unsigned int i = 0; i < rank_local_obstacles; ++i)
        {
          Handle              obstacle_handle = properties_global_obstacles.register_particle();
          dealii::Point<dim>  obstacle_location;
          std::vector<number> obstacle_properties(ObstacleType::n_obstacle_properties);
          if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == rank)
            {
              dealii::Particles::ParticleAccessor<dim> local_obstacle =
                *(std::next(obstacle_handler.begin(), i));
              obstacle_location                          = local_obstacle.get_location();
              dealii::ArrayView<number> local_properties = local_obstacle.get_properties();
              local_properties[ObstacleType::Properties::particle_id] = local_obstacle.get_id();
              for (unsigned int j = 0; j < local_properties.size(); ++j)
                obstacle_properties[j] = local_properties[j];
            }
          // broadcast particle location
          for (int d = 0; d < dim; ++d)
            {
              number location = obstacle_location[d];
              dealii::Utilities::MPI::broadcast(&location, 1, rank, mpi_communicator);
              obstacle_location[d] = location;
            }
          properties_global_obstacles.set_location(obstacle_handle, obstacle_location);
          // broadcast particle properties
          dealii::Utilities::MPI::broadcast(obstacle_properties.data(),
                                            obstacle_properties.size(),
                                            rank,
                                            mpi_communicator);
          dealii::ArrayView<number> local_properties =
            properties_global_obstacles.get_properties(obstacle_handle);
          for (unsigned int j = 0; j < local_properties.size(); ++j)
            local_properties[j] = obstacle_properties[j];
        }
    }
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::deregister_property_pool()
  const
{
  // The property pool containing the properties of all global particles is initialized once during
  // the `reinit()` call and remains unchanged thereafter. As a result, we don't need to track
  // individual handles explicitly. We know that handles are assigned sequentially, starting from
  // zero up to the number of registered slots. Therefore, a simple loop is sufficient to deregister
  // all particles before releasing the associated resources.
  for (unsigned int i = 0; i < properties_global_obstacles.n_registered_slots(); ++i)
    properties_global_obstacles.deregister_particle(i);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::insert_global_particles(
  const std::vector<dealii::Point<dim, number>> &obstacle_locations,
  const std::vector<std::vector<number>>        &obstacle_properties)
{
  Assert(obstacle_locations.size() == obstacle_properties.size(),
         dealii::ExcMessage(
           "The number of obstacle locations and obstacle properties must be the same."));

  std::vector<dealii::BoundingBox<dim>> local_bounding_box =
    dealii::GridTools::compute_mesh_predicate_bounding_box(
      obstacle_handler.get_triangulation(), dealii::IteratorFilters::LocallyOwnedCell());
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

  reinit();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::deserialize()
{
  obstacle_handler.deserialize();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::prepare_for_serialization()
{
  obstacle_handler.prepare_for_serialization();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::
  prepare_for_coarsening_and_refinement()
{
  obstacle_handler.prepare_for_coarsening_and_refinement();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::
  unpack_after_coarsening_and_refinement()
{
  obstacle_handler.unpack_after_coarsening_and_refinement();
  obstacle_handler.sort_particles_into_subdomains_and_cells();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::register_particle_output(
  Postprocessor<dim, number> &postprocessor) const
{
  const auto [property_names, property_component_interpretations] =
    ObstacleType::get_property_names_and_component_interpretation();

  postprocessor.register_obstacle_output(&obstacle_handler,
                                         property_names,
                                         property_component_interpretations);
}

template <int dim, typename number, typename ObstacleType>
unsigned int
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::n_global_particles() const
{
  return obstacle_handler.n_global_particles();
}

template <int dim, typename number, typename ObstacleType>
unsigned int
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::n_locally_owned_particles()
  const
{
  return obstacle_handler.n_locally_owned_particles();
}

template <int dim, typename number, typename ObstacleType>
typename std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::locally_owned_particle_range()
  const
{
  return std::ranges::subrange<ParticleIterator<dim, number>>(
    ParticleIterator<dim, number>(obstacle_handler.begin()),
    ParticleIterator<dim, number>(obstacle_handler.end()));
}

template <int dim, typename number, typename ObstacleType>
typename std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::particles_in_cell(
  typename dealii::Triangulation<dim>::active_cell_iterator cell) const
{
  return std::ranges::subrange<ParticleIterator<dim, number>>(
    ParticleIterator<dim, number>(obstacle_handler.particles_in_cell(cell).begin()),
    ParticleIterator<dim, number>(obstacle_handler.particles_in_cell(cell).end()));
}


template <int dim, typename number, typename ObstacleType>
dealii::Particles::PropertyPool<dim> &
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::
  get_global_particle_properties()
{
  return properties_global_obstacles;
}

template <int dim, typename number, typename ObstacleType>
const dealii::Particles::PropertyPool<dim> &
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::
  get_global_particle_properties() const
{
  return properties_global_obstacles;
}

template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<3, double, MeltPoolDG::SphericalParticle<3, double>>;
