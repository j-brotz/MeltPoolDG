#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/property_pool.h>

#include "meltpooldg/particles/particle_accessor.hpp"
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_iterator.hpp>

#include <boost/container/small_vector.hpp>

#include <memory>
#include <vector>

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::ObstacleCompleteDomainSearch(
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping,
  dealii::TimerOutput              &timer)
  : obstacle_handler(std::make_unique<dealii::Particles::ParticleHandler<dim>>(
      triangulation,
      mapping,
      ObstacleType::n_obstacle_properties))
  , properties_global_obstacles(
      std::make_unique<dealii::Particles::PropertyPool<dim>>(ObstacleType::n_obstacle_properties))
  , timer(timer)
  , triangulation(&triangulation)
  , level_cell_partitioner(triangulation, level_to_store_particles)
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
  for (unsigned int level = 0; level < obstacle_handler->get_triangulation().n_global_levels();
       ++level)
    {
      if (triangulation->begin(level)->minimum_vertex_distance() < max_particle_radius)
        {
          level_to_store_particles = level == 0 ? 0 : level - 1;
          break;
        }
    }

  deregister_property_pool();
  properties_global_obstacles->clear();
  broadcast_global_particles();

  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles->n_registered_slots();
       ++src_handle)
    {
      DEMParticleAccessor<dim, number> particle(*properties_global_obstacles, src_handle);
    }

  level_cell_partitioner.reinit();
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim> &dst,
  const dealii::CellAccessor<dim>      &cell) const
{
  dealii::TimerOutput::Scope t(timer, "particles in cell search");

  std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> handles;
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles->n_registered_slots();
       ++src_handle)
    {
      if (ObstacleType::is_in_cell(*properties_global_obstacles, src_handle, cell))
        {
          auto dst_handle = dst.register_particle();
          handles.emplace_back(dst_handle);
          dst.set_location(dst_handle, properties_global_obstacles->get_location(src_handle));
          auto dst_properties = dst.get_properties(dst_handle);
          auto src_properties = properties_global_obstacles->get_properties(src_handle);

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
  dealii::TimerOutput::Scope t(timer, "particles in cell search");

  std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> handles;
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles->n_registered_slots();
       ++src_handle)
    {
      for (const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell : cells)
        {
          if (ObstacleType::is_in_cell(*properties_global_obstacles, src_handle, *cell))
            {
              auto dst_handle = dst.register_particle();
              handles.emplace_back(dst_handle);
              dst.set_location(dst_handle, properties_global_obstacles->get_location(src_handle));
              auto dst_properties = dst.get_properties(dst_handle);
              auto src_properties = properties_global_obstacles->get_properties(src_handle);

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
boost::container::small_vector<MeltPoolDG::DEMParticleAccessor<dim, number>, 3 * dim>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::contact_particles(
  const DEMParticleAccessor<dim, number> &particle,
  const number                            relative_tolerance) const
{
  dealii::TimerOutput::Scope t(timer, "contact particle search");

  // We assume the max number of contacts per particle to be 3*dim, which is a reasonable
  // assumption for spherical particles including some tolerance offset for the contact
  // distance. This allows us to use a small_vector for efficient storage without dynamic memory
  // allocation in most cases.
  boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim> contacts;

  for (const auto &other : std::ranges::subrange<ParticleIterator<dim, number>>(
         ParticleIterator<dim, number>(*properties_global_obstacles, 0),
         ParticleIterator<dim, number>(*properties_global_obstacles,
                                       properties_global_obstacles->n_registered_slots())))
    {
      if ((particle.get_location() - other.get_location()).norm_square() <
            dealii::Utilities::fixed_power<2>((other.radius() + particle.radius()) *
                                              (1. + relative_tolerance)) and
          particle.id() != other.id())
        contacts.push_back(other);
    }

  return contacts;
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::broadcast_global_particles()
  const
{
  dealii::TimerOutput::Scope t(timer, "global particle broadcast");

  deregister_property_pool();
  properties_global_obstacles->clear();
  using Handle = typename dealii::Particles::PropertyPool<dim>::Handle;
  for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
       ++rank)
    {
      unsigned int rank_local_obstacles = obstacle_handler->n_locally_owned_particles();
      dealii::Utilities::MPI::broadcast(&rank_local_obstacles, 1, rank, mpi_communicator);
      for (unsigned int i = 0; i < rank_local_obstacles; ++i)
        {
          Handle              obstacle_handle = properties_global_obstacles->register_particle();
          dealii::Point<dim>  obstacle_location;
          std::vector<number> obstacle_properties(ObstacleType::n_obstacle_properties);
          if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == rank)
            {
              dealii::Particles::ParticleAccessor<dim> local_obstacle =
                *(std::next(obstacle_handler->begin(), i));
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
          properties_global_obstacles->set_location(obstacle_handle, obstacle_location);
          // broadcast particle properties
          dealii::Utilities::MPI::broadcast(obstacle_properties.data(),
                                            obstacle_properties.size(),
                                            rank,
                                            mpi_communicator);
          dealii::ArrayView<number> local_properties =
            properties_global_obstacles->get_properties(obstacle_handle);
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
  if (properties_global_obstacles == nullptr)
    return;

  for (unsigned int i = 0; i < properties_global_obstacles->n_registered_slots(); ++i)
    properties_global_obstacles->deregister_particle(i);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::
  sort_particles_into_local_level_cells()
{
  cell_to_locally_owned_particle_cache.clear();
  for (dealii::Particles::ParticleIterator<dim> particle = obstacle_handler->begin();
       particle != obstacle_handler->end();
       ++particle)
    {
      if (particle->get_surrounding_cell()->is_locally_owned())
        {
          typename dealii::Triangulation<dim>::cell_iterator cell =
            particle->get_surrounding_cell();
          while (cell->level() > level_to_store_particles)
            {
              cell = cell->parent();
            }
          cell_to_locally_owned_particle_cache[cell].push_back(particle);
        }
    }
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::communicate_ghost_particles()
{
  cell_to_ghost_particle_cache.clear();

  // Step 1: Send the number of particles to be sent to each rank
  std::vector<MPI_Request>          send_requests(level_cell_partitioner.n_processes_to_send_to());
  unsigned                          request_send_index = 0;
  std::unordered_map<int, unsigned> n_particles_to_send(
    level_cell_partitioner.n_processes_to_send_to());
  for (const auto &[rank, cells] : level_cell_partitioner.get_cell_to_rank_send())
    {
      n_particles_to_send[rank] = 0;
      for (const dealii::CellId &cell_id : cells)
        {
          n_particles_to_send[rank] +=
            cell_to_locally_owned_particle_cache[triangulation->create_cell_iterator(cell_id)]
              .size();
        }
      MPI_Isend(&n_particles_to_send[rank],
                1,
                MPI_UNSIGNED,
                rank,
                0,
                mpi_communicator,
                &send_requests[request_send_index++]);
    }

  // Step 2: Receive the number of particles to be received from each rank
  std::vector<MPI_Request> receive_requests(level_cell_partitioner.n_processes_to_receive_from());
  unsigned                 request_receive_index = 0;

  // First: Rank to receive from; second: number of particles to receive from that rank
  std::vector<std::pair<int, unsigned>> n_particles_to_receive(
    level_cell_partitioner.n_processes_to_receive_from());
  for (const int rank : level_cell_partitioner.get_particle_receiver_ranks())
    {
      n_particles_to_receive[request_receive_index].first = rank;
      MPI_Irecv(&n_particles_to_receive[request_receive_index].second,
                1,
                MPI_UNSIGNED,
                rank,
                0,
                mpi_communicator,
                &receive_requests[request_receive_index]);
      request_receive_index++;
    }

  MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(receive_requests.size(), receive_requests.data(), MPI_STATUSES_IGNORE);

  // Step 3: Send the particle data to the respective ranks
  std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
  for (const auto &[rank, cells] : level_cell_partitioner.get_cell_to_rank_send())
    {
      std::vector<char> send_buffer(n_particles_to_send[rank] *
                                    serialized_size_in_bytes(ObstacleType::n_obstacle_properties));

      unsigned iter = 0;
      for (const dealii::CellId &cell_id : cells)
        {
          for (dealii::Particles::ParticleIterator<dim> particle :
               cell_to_locally_owned_particle_cache[triangulation->create_cell_iterator(cell_id)])
            {
              write_particle_data_to_memory(send_buffer.data() +
                                              iter * serialized_size_in_bytes(
                                                       ObstacleType::n_obstacle_properties),
                                            particle,
                                            triangulation->create_cell_iterator(cell_id));

              send_futures.push_back(dealii::Utilities::MPI::isend(
                send_buffer,
                obstacle_handler->get_triangulation().get_mpi_communicator(),
                rank,
                1));

              iter++;
            }
        }
    }

  // Step 4: Receive the particle data from the respective ranks and populate the ghost particle
  // cache
  std::vector<dealii::Utilities::MPI::Future<std::vector<char>>> recv_futures;
  recv_futures.reserve(n_particles_to_receive.size());
  for (const auto &[rank, n_particles] : n_particles_to_receive)
    {
      recv_futures.push_back(dealii::Utilities::MPI::irecv<std::vector<char>>(
        obstacle_handler->get_triangulation().get_mpi_communicator(), rank, 1));
    }

  for (auto &future : recv_futures)
    {
      future.wait();
      ReceivedParticleData received_data =
        read_particle_data_from_memory(future.get().data(),
                                       *properties_global_obstacles,
                                       ObstacleType::n_obstacle_properties);
      typename dealii::Triangulation<dim>::cell_iterator cell(
        &obstacle_handler->get_triangulation(), received_data.cell_level, received_data.cell_index);

      cell_to_ghost_particle_cache[cell].push_back(received_data.handle);
    }

  for (auto &future : send_futures)
    {
      future.wait();
    }
}

template <int dim, typename number, typename ObstacleType>
void *
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::write_particle_data_to_memory(
  void                                                     *data_pointer,
  const dealii::Particles::ParticleIterator<dim>            particle,
  const typename dealii::Triangulation<dim>::cell_iterator &cell) const
{
  // TODO: Do we need to consider memory alignment issues here?
  dealii::types::particle_index *id_data =
    static_cast<dealii::types::particle_index *>(data_pointer);
  *id_data = particle->get_id();
  ++id_data;

  int *cell_data = reinterpret_cast<int *>(id_data);
  *cell_data     = cell->level();
  ++cell_data;
  *cell_data = cell->index();
  ++cell_data;

  double *pdata = reinterpret_cast<double *>(cell_data);

  // Write location
  for (unsigned int i = 0; i < dim; ++i, ++pdata)
    *pdata = particle->get_location()[i];

  // Write properties
  if (particle->has_properties())
    {
      dealii::ArrayView<const double> particle_properties = particle->get_properties();
      for (unsigned int i = 0; i < particle_properties.size(); ++i, ++pdata)
        *pdata = particle_properties[i];
    }

  return static_cast<void *>(pdata);
}

template <int dim, typename number, typename ObstacleType>
auto
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::read_particle_data_from_memory(
  void                                 *data_pointer,
  dealii::Particles::PropertyPool<dim> &property_pool,
  const unsigned                        n_properties) const -> ReceivedParticleData
{
  ReceivedParticleData received_data;
  received_data.handle = property_pool.register_particle();

  const dealii::types::particle_index *id_data =
    static_cast<const dealii::types::particle_index *>(data_pointer);
  property_pool.set_id(received_data.handle, *id_data++);

  const int *cell_data     = reinterpret_cast<const int *>(id_data);
  received_data.cell_level = *cell_data++;
  received_data.cell_index = *cell_data++;

  const double *pdata = reinterpret_cast<const double *>(id_data);

  dealii::Point<dim> location;
  for (unsigned int i = 0; i < dim; ++i)
    location[i] = *pdata++;
  property_pool.set_location(received_data.handle, location);

  if (n_properties > 0)
    {
      const dealii::ArrayView<double> particle_properties =
        property_pool.get_properties(received_data.handle);
      for (unsigned int i = 0; i < n_properties; ++i)
        particle_properties[i] = *pdata++;
    }

  return received_data;
}

template <int dim, typename number, typename ObstacleType>
std::size_t
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::serialized_size_in_bytes(
  unsigned int n_properties) const
{
  return sizeof(dealii::types::particle_index) + 2 * sizeof(int) + dim * sizeof(double) +
         n_properties * sizeof(double);
}

template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<3, double, MeltPoolDG::SphericalParticle<3, double>>;