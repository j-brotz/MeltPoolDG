#include <deal.II/base/exception_macros.h>
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

#include <algorithm>
#include <memory>
#include <vector>

#include "mpi.h"

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
  , level_cell_partitioner(triangulation)
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
  Assert(triangulation != nullptr, dealii::ExcMessage("Triangulation pointer is null."));

  if (max_particle_radius == 0)
    {
      // If there are only particles with zero radius, we can store them on the finest level.
      level_to_store_particles = triangulation->n_global_levels() - 1;
    }
  else
    {
      // Determine the appropriate level to store particles based on the maximum particle radius and
      // the minimum vertex distance of cells on each level. We want to ensure that particles are
      // stored in cells that are sufficiently large to contain them, which typically means that the
      // minimum vertex distance should be at least twice the maximum particle radius.
      // For the case of contact search, this guarantees that all potential contact partners of a
      // particle are located in the same cell or in the neighboring cells. If we are only
      // interested in getting the active cells occupied by a particle, a cell size a cell size of
      // the maximum particle radius would be sufficient. However, to keep things simple we use the
      // same level for both purposes for now.
      for (unsigned int level = 0; level < obstacle_handler->get_triangulation().n_global_levels();
           ++level)
        {
          if (triangulation->begin(level)->minimum_vertex_distance() < 2 * max_particle_radius)
            {
              level_to_store_particles = level == 0 ? 0 : level - 1;
              break;
            }
        }
    }

  MPI_Allreduce(MPI_IN_PLACE, &level_to_store_particles, 1, MPI_INT, MPI_MIN, mpi_communicator);

  deregister_property_pool();
  properties_global_obstacles->clear();

  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles->n_registered_slots();
       ++src_handle)
    {
      DEMParticleAccessor<dim, number> particle(*properties_global_obstacles, src_handle, true);
    }

  level_cell_cache.reinit(*triangulation, level_to_store_particles);
  level_cell_partitioner.build_pattern(level_to_store_particles);
  sort_particles_into_subdomains_and_cells();
}

template <int dim, typename number, typename ObstacleType>
boost::container::small_vector<MeltPoolDG::DEMParticleAccessor<dim, number>, 3 * dim>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::contact_particles(
  const DEMParticleAccessor<dim, number> &particle,
  const number                            relative_tolerance) const
{
  // We assume the max number of contacts per particle to be 3*dim, which is a reasonable
  // assumption for spherical particles including some tolerance offset for the contact
  // distance. This allows us to use a small_vector for efficient storage without dynamic memory
  // allocation in most cases.
  boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim> contacts;

  dealii::TriaIterator<dealii::CellAccessor<dim>> cell =
    find_particle_storage_cell(particle.get_surrounding_cell());

  for (const auto &other : find_relevant_particles(cell))
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
{}

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
  obstacle_handler->sort_particles_into_subdomains_and_cells();
  cell_to_locally_owned_particle_cache.clear();
  cell_to_locally_owned_particle_cache.resize(triangulation->n_cells(level_to_store_particles));
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
          cell_to_locally_owned_particle_cache[cell->index()].push_back(particle);
        }
    }
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::communicate_ghost_particles()
{
  cell_to_ghost_particle_cache.clear();
  cell_to_ghost_particle_cache.resize(triangulation->n_cells(level_to_store_particles));
  rank_to_handle.clear();
  rank_to_handle.resize(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator));
  rank_to_n_ghost_particles.clear();
  rank_to_n_ghost_particles.resize(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator), 0);
  deregister_property_pool();

  // Step 1: Send the number of particles to be sent to each rank
  std::vector<MPI_Request>          send_requests(level_cell_partitioner.n_processes_to_send_to());
  unsigned                          request_send_index = 0;
  std::unordered_map<int, unsigned> n_particles_to_send(
    level_cell_partitioner.n_processes_to_send_to());
  for (const auto &[rank, cells] : level_cell_partitioner.cells_to_send())
    {
      n_particles_to_send[rank] = 0;
      for (const dealii::CellId &cell_id : cells)
        {
          n_particles_to_send[rank] +=
            cell_to_locally_owned_particle_cache[triangulation->create_cell_iterator(cell_id)
                                                   ->index()]
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
  for (const int rank : level_cell_partitioner.receive_ranks())
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

  // Fill the rank_to_n_ghost_particles map for caching purposes
  for (const auto &[rank, n_particles] : n_particles_to_send)
    {
      rank_to_n_ghost_particles[rank] = n_particles;
    }

  // Step 3: Send the particle data to the respective ranks
  std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
  for (const auto &[rank, cells] : level_cell_partitioner.cells_to_send())
    {
      std::vector<char> send_buffer(n_particles_to_send[rank] *
                                    serialized_size_in_bytes(ObstacleType::n_obstacle_properties));

      unsigned iter = 0;
      for (const dealii::CellId &cell_id : cells)
        {
          for (dealii::Particles::ParticleIterator<dim> particle :
               cell_to_locally_owned_particle_cache[triangulation->create_cell_iterator(cell_id)
                                                      ->index()])
            {
              write_particle_data_to_memory(send_buffer.data() +
                                              iter * serialized_size_in_bytes(
                                                       ObstacleType::n_obstacle_properties),
                                            particle,
                                            triangulation->create_cell_iterator(cell_id));
              iter++;
            }
        }

      send_futures.push_back(dealii::Utilities::MPI::isend(
        send_buffer, obstacle_handler->get_triangulation().get_mpi_communicator(), rank, 1));
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

  for (unsigned int i = 0; i < recv_futures.size(); ++i)
    {
      auto &future = recv_futures[i];
      future.wait();
      // Note: recv_futures and n_particles_to_receive have the same order, so we can use the same
      // index to access both
      std::vector<char> recv_buffer = future.get();
      for (unsigned int p = 0; p < n_particles_to_receive[i].second; ++p)
        {
          ReceivedParticleData received_data = read_particle_data_from_memory(
            recv_buffer.data() + p * serialized_size_in_bytes(ObstacleType::n_obstacle_properties),
            *properties_global_obstacles,
            ObstacleType::n_obstacle_properties);

          typename dealii::Triangulation<dim>::cell_iterator cell;
          if (triangulation->contains_cell(received_data.cell_id))
            cell = triangulation->create_cell_iterator(received_data.cell_id);
          else
            {
              auto child_indices = received_data.cell_id.get_child_indices();
              auto new_cell_id   = dealii::CellId(received_data.cell_id.get_coarse_cell_id(),
                                                child_indices.size() - 1,
                                                child_indices.data());
              cell               = triangulation->create_cell_iterator(new_cell_id);
            }

          cell_to_ghost_particle_cache[cell->index()].push_back(received_data.handle);

          // TODO: We assume that the index i is the same for both the recv_futures and the
          // corresponding rank in the n_particles_to_receive vector, which should be the case since
          // they are filled in the same order. However, it might be safer to explicitly find the
          // corresponding rank for the current future to avoid any potential mismatches.
          rank_to_handle[n_particles_to_receive[i].first].push_back(received_data.handle);
        }
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

  unsigned int                              *cell_data = reinterpret_cast<unsigned int *>(id_data);
  const typename dealii::CellId::binary_type cell_id_binary = cell->id().template to_binary<dim>();
  std::ranges::copy(cell_id_binary, cell_data);
  *cell_data = *cell->id().template to_binary<dim>().data();
  cell_data += cell->id().template to_binary<dim>().size();

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

  const unsigned int                  *cell_data = reinterpret_cast<const unsigned int *>(id_data);
  typename dealii::CellId::binary_type cell_id_binary; // same as std::array<unsigned int, 4>
  std::copy(cell_data, cell_data + cell_id_binary.size(), cell_id_binary.data());
  dealii::CellId cell_id(cell_id_binary);
  received_data.cell_id = cell_id;
  cell_data += cell_id_binary.size();

  const double *pdata = reinterpret_cast<const double *>(cell_data);

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
  return sizeof(dealii::types::particle_index) + 4 * sizeof(unsigned int) + dim * sizeof(double) +
         n_properties * sizeof(double);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::compress()
{
  using MPIExchangeType = std::vector<std::pair<unsigned int, std::vector<number>>>;

  // We only want to compress the force and torque properties
  std::array<unsigned, dim + axial_dim<dim>> property_indices_to_compress = {};
  int                                        i                            = 0;
  for (unsigned int d = 0; d < dim; ++d)
    property_indices_to_compress[i++] = ObstacleType::Properties::force + d;

  for (unsigned int d = 0; d < axial_dim<dim>; ++d)
    property_indices_to_compress[i++] = ObstacleType::Properties::torque + d;

  // Send ghost particle properties which shall be compressed to the owning rank of the
  // particles.
  std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
  send_futures.reserve(rank_to_handle.size());

  for (unsigned int target_rank = 0; target_rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++target_rank)
    {
      if (rank_to_handle[target_rank].empty())
        continue;

      MPIExchangeType ghost_properties_to_send;
      ghost_properties_to_send.reserve(rank_to_handle[target_rank].size() * property_indices_to_compress.size());
      for (const auto &handle : rank_to_handle[target_rank])
        {
          dealii::ArrayView<double> property_values =
            properties_global_obstacles->get_properties(handle);

          auto send_view = property_indices_to_compress |
                           std::views::transform([&property_values](unsigned int property_index) {
                             return property_values[property_index];
                           });

          ghost_properties_to_send.emplace_back(properties_global_obstacles->get_id(handle),
                                                std::vector<number>(send_view.begin(),
                                                                    send_view.end()));
        }
      send_futures.emplace_back(
        dealii::Utilities::MPI::isend(ghost_properties_to_send, mpi_communicator, target_rank, 0));
    }

  // Receive compressed properties for ghost particles from other ranks and update the local
  // property pool accordingly.
  std::vector<dealii::Utilities::MPI::Future<MPIExchangeType>> receive_futures;

  
  for (unsigned int rank = 0; rank < rank_to_n_ghost_particles.size(); ++rank)
    {
      if (rank_to_n_ghost_particles[rank] > 0)
        {
          receive_futures.emplace_back(
            dealii::Utilities::MPI::irecv<MPIExchangeType>(mpi_communicator, rank, 0));
        }
    }

  // Wait for all sends and receives to complete, and update the local property pool with the
  // received compressed properties.
  for (auto &future : receive_futures)
    {
      future.wait();
      MPIExchangeType received_properties = future.get();
      for (const auto &[particle_id, other_values] : received_properties)
        {
          Assert(
            particle_id_to_iterator_cache.contains(particle_id),
            dealii::ExcMessage(
              "Received compressed properties for a particle that is not in the local property pool."));

          dealii::ArrayView<double> property_values =
            particle_id_to_iterator_cache[particle_id]->get_properties();

          unsigned int i = 0;
          for (const unsigned int property_index : property_indices_to_compress)
            {
              property_values[property_index] += other_values[i++];
            }
        }
    }

  for (auto &future : send_futures)
    {
      future.wait();
    }
}

template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<3, double, MeltPoolDG::SphericalParticle<3, double>>;