#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>

template <int dim, typename number>
template <typename ObstacleDataStructureType>
MeltPoolDG::ObstacleDataStructure<dim, number>::ObstacleDataStructureModel<
  ObstacleDataStructureType>::ObstacleDataStructureModel(ObstacleDataStructureType
                                                           &&obstacle_data_structure)
  : obstacle_data_structure(std::move(obstacle_data_structure))
{}

template <int dim, typename number>
template <typename ObstacleDataStructureType>
void
MeltPoolDG::ObstacleDataStructure<dim, number>::ObstacleDataStructureModel<
  ObstacleDataStructureType>::reinit()
{
  obstacle_data_structure.reinit();
}

template <int dim, typename number>
template <typename ObstacleDataStructureType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleDataStructure<dim, number>::ObstacleDataStructureModel<
  ObstacleDataStructureType>::get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                                                    const dealii::CellAccessor<dim> &cell) const
{
  return obstacle_data_structure.get_obstacles_in_cell(dst, cell);
}

template <int dim, typename number>
template <typename ObstacleDataStructureType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleDataStructure<dim, number>::ObstacleDataStructureModel<
  ObstacleDataStructureType>::get_obstacles_in_cell_batch(dealii::Particles::PropertyPool<dim> &dst,
                                                          const dealii::MatrixFree<dim, number>
                                                                            &matrix_free,
                                                          const unsigned int cell_batch_id) const
{
  return obstacle_data_structure.get_obstacles_in_cell_batch(dst, matrix_free, cell_batch_id);
}

template <int dim, typename number>
template <typename ObstacleDataStructureType>
MeltPoolDG::ObstacleDataStructure<dim, number>::ObstacleDataStructure(
  ObstacleDataStructureType &&obstacle_data_structure)
  : obstacle_data_structure_pimpl(
      std::make_unique<ObstacleDataStructureModel<ObstacleDataStructureType>>(
        std::move(obstacle_data_structure)))
{}

template <int dim, typename number>
void
MeltPoolDG::ObstacleDataStructure<dim, number>::reinit()
{
  obstacle_data_structure_pimpl->reinit();
}

template <int dim, typename number>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleDataStructure<dim, number>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim> &dst,
  const dealii::CellAccessor<dim>      &cell) const
{
  return obstacle_data_structure_pimpl->get_obstacles_in_cell(dst, cell);
}

template <int dim, typename number>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleDataStructure<dim, number>::get_obstacles_in_cell_batch(
  dealii::Particles::PropertyPool<dim>  &dst,
  const dealii::MatrixFree<dim, number> &matrix_free,
  const unsigned int                     cell_batch_id) const
{
  return obstacle_data_structure_pimpl->get_obstacles_in_cell_batch(dst,
                                                                    matrix_free,
                                                                    cell_batch_id);
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::ObstacleCompleteDomainSearch(
  const dealii::Particles::ParticleHandler<dim> &obstacle_handler)
  : obstacle_handler(obstacle_handler)
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
bool
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::process_particle_in_cell(
  const typename dealii::Particles::PropertyPool<dim>::Handle        &src_handle,
  const dealii::CellAccessor<dim>                                    &cell,
  dealii::Particles::PropertyPool<dim>                               &dst,
  std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> &target_handles) const
{
  if (ObstacleType::is_in_cell(properties_global_obstacles, src_handle, cell))
    {
      auto dst_handle = dst.register_particle();
      target_handles.emplace_back(dst_handle);
      dst.set_location(dst_handle, properties_global_obstacles.get_location(src_handle));
      auto dst_properties = dst.get_properties(dst_handle);
      auto src_properties = properties_global_obstacles.get_properties(src_handle);

      for (unsigned int n_property = 0; n_property < ObstacleType::n_obstacle_properties;
           ++n_property)
        {
          dst_properties[n_property] = src_properties[n_property];
        }
      return true;
    }
  return false;
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
      process_particle_in_cell(src_handle, cell, dst, handles);
    }
  return handles;
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::get_obstacles_in_cell_batch(
  dealii::Particles::PropertyPool<dim>  &dst,
  const dealii::MatrixFree<dim, number> &matrix_free,
  const unsigned int                     cell_batch_id) const
{
  std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> handles;
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles.n_registered_slots();
       ++src_handle)
    {
      for (unsigned int batch_lane = 0;
           batch_lane < matrix_free.n_active_entries_per_cell_batch(cell_batch_id);
           ++batch_lane)
        {
          if (process_particle_in_cell(src_handle,
                                       *matrix_free.get_cell_iterator(cell_batch_id, batch_lane),
                                       dst,
                                       handles))
            {
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


template struct MeltPoolDG::ObstacleDataStructure<1, double>;
template struct MeltPoolDG::ObstacleDataStructure<2, double>;
template struct MeltPoolDG::ObstacleDataStructure<3, double>;

template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<3, double, MeltPoolDG::SphericalParticle<3, double>>;
