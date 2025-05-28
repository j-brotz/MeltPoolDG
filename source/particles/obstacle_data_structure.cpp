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
void
MeltPoolDG::ObstacleDataStructure<dim, number>::ObstacleDataStructureModel<
  ObstacleDataStructureType>::get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                                                    const dealii::CellAccessor<dim> &cell) const
{
  return obstacle_data_structure.compute_force_on_obstacle(dst, cell);
}

template <int dim, typename number>
template <typename ObstacleDataStructureType>
void
MeltPoolDG::ObstacleDataStructure<dim, number>::ObstacleDataStructureModel<
  ObstacleDataStructureType>::get_obstacles_in_cell_batch(dealii::Particles::PropertyPool<dim> &dst,
                                                          const dealii::MatrixFree<dim, number>
                                                                            &matrix_free,
                                                          const unsigned int cell_batch_id,
                                                          const unsigned int n_lanes) const
{
  return obstacle_data_structure.compute_force_on_obstacle(dst,
                                                           matrix_free,
                                                           cell_batch_id,
                                                           n_lanes);
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
void
MeltPoolDG::ObstacleDataStructure<dim, number>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim> &dst,
  const dealii::CellAccessor<dim>      &cell) const
{
  return obstacle_data_structure_pimpl->get_obstacles_in_cell(dst, cell);
}

template <int dim, typename number>
void
MeltPoolDG::ObstacleDataStructure<dim, number>::get_obstacles_in_cell_batch(
  dealii::Particles::PropertyPool<dim>  &dst,
  const dealii::MatrixFree<dim, number> &matrix_free,
  const unsigned int                     cell_batch_id,
  const unsigned int                     n_lanes) const
{
  return obstacle_data_structure_pimpl->get_obstacles_in_cell_batch(dst,
                                                                    matrix_free,
                                                                    cell_batch_id,
                                                                    n_lanes);
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::ObstacleCompleteDomainSearch(
  const dealii::Particles::ParticleHandler<dim> &obstacle_handler)
  : obstacle_handler(obstacle_handler)
  , properties_global_obstacles(ObstacleType::n_obstacle_properties)
{}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::reinit() const
{
  for (unsigned int i = 0; i < properties_global_obstacles.n_registered_slots(); ++i)
    {
      properties_global_obstacles.deregister_particle(i);
    }
  broadcast_global_particles();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim> &dst,
  const dealii::CellAccessor<dim>      &cell) const
{
  dst.clear();
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles.n_registered_slots();
       ++src_handle)
    {
      if (ObstacleType::is_in_cell(properties_global_obstacles, src_handle, cell))
        {
          auto dst_handle = dst.register_particle();
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

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::get_obstacles_in_cell_batch(
  dealii::Particles::PropertyPool<dim>  &dst,
  const dealii::MatrixFree<dim, number> &matrix_free,
  const unsigned int                     cell_batch_id,
  const unsigned int                     n_lanes) const
{
  dst.clear();
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles.n_registered_slots();
       ++src_handle)
    {
      for (unsigned int batch_lane = 0; batch_lane < n_lanes; ++batch_lane)
        {
          if (ObstacleType::is_in_cell(properties_global_obstacles,
                                       src_handle,
                                       *matrix_free.get_cell_iterator(cell_batch_id, batch_lane)))
            {
              auto dst_handle = dst.register_particle();
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
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::broadcast_global_particles()
  const
{
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

template class MeltPoolDG::ObstacleDataStructure<1, double>;
template class MeltPoolDG::ObstacleDataStructure<2, double>;
template class MeltPoolDG::ObstacleDataStructure<3, double>;

template class MeltPoolDG::
  ObstacleCompleteDomainSearch<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::
  ObstacleCompleteDomainSearch<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::
  ObstacleCompleteDomainSearch<3, double, MeltPoolDG::SphericalParticle<3, double>>;
