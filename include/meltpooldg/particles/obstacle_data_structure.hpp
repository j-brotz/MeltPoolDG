#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/property_pool.h>

#include "meltpooldg/particles/particle.hpp"
#include "meltpooldg/particles/particle_accessor.hpp"
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/particle_iterator.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>

#include <boost/container/small_vector.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <vector>

#include "mpi.h"

namespace MeltPoolDG
{
  /**
   * @brief A simple search utility that performs a linear scan over all particles.
   *
   * This struct implements a very basic search algorithm by iterating over all particles
   * in the domain and individually checking whether each one satisfies the desired condition.
   *
   * @note This approach is not optimized for performance and is intended primarily for
   * debugging, prototyping, or use in small-scale simulations.
   */
  template <int dim,
            typename number,
            typename ObstacleType = MeltPoolDG::SphericalParticle<dim, number>>
  struct CellListParticleHandler
  {
  public:
    static constexpr std::size_t max_particles_per_active_cell = 20;
    using particle_accessor                                    = DEMParticleAccessor<dim, number>;

    CellListParticleHandler(const dealii::Triangulation<dim> &triangulation,
                            const dealii::Mapping<dim>       &mapping,
                            dealii::TimerOutput              &timer);

    /**
     * @brief Destructor. Explicitly deregisters all particles from the global obstacle property
     * pool.
     */
    ~CellListParticleHandler();

    CellListParticleHandler(CellListParticleHandler &&) = default;

    CellListParticleHandler &
    operator=(CellListParticleHandler &&) = default;

    /**
     * @brief Reinitializes the internal data structure by synchronizing obstacle data across all
     * MPI processes.
     *
     * This function communicates all locally owned obstacles to every other process in the MPI
     * communicator. As a result, each process obtains and stores a complete local copy of all
     * obstacles, regardless of ownership. This enables fully local access to obstacle data
     * during subsequent computations.
     */
    void
    reinit(number max_particle_influence_radius);

    /**
     * @brief Identify obstacles that partially or fully occupy the specified cell, and store their
     * properties in the destination property pool.
     *
     * This function scans a cell and collects the properties of all obstacles that intersect
     * with the cell. These properties are then stored in the provided destination property
     * pool.
     *
     * The identification strategy is based on a brute-force approach: the function iterates
     * over all globally known obstacles (previously synchronized), checks each one for
     * relevance to the specified cell, and includes it if applicable.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param cell The cell of interest.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    template <typename CellContainer, typename ObstacleContainer>
    void
    get_obstacles_in_cell(const CellContainer &cells, ObstacleContainer &particles) const
    {
      for (const auto &cell : cells)
        {
          std::size_t current_size = particles.size();

          find_relevant_particles(find_particle_storage_cell(cell), particles);
          auto old_end   = particles.begin() + current_size;
          auto write_ptr = old_end;
          for (auto read_ptr = old_end; read_ptr != particles.end(); ++read_ptr)
            {
              auto found_it =
                std::find_if(particles.begin(), old_end, [&read_ptr](const auto &particle) {
                  return particle.id() == read_ptr->id();
                });
              if (found_it == old_end)
                {
                  if (write_ptr->id() != read_ptr->id())
                    *write_ptr = std::move(*read_ptr);
                  ++write_ptr;
                }
            }
          particles.erase(write_ptr, particles.end());
        }
    }

    template <typename ObstacleContainer>
    void
    get_obstacles_in_cell(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell,
                          ObstacleContainer                                     &particles) const
    {
      find_relevant_particles(find_particle_storage_cell(cell), particles);
    }

    dealii::TriaIterator<dealii::CellAccessor<dim>>
    find_particle_storage_cell(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) const
    {
      Assert(
        cell->level() >= level_to_store_particles,
        dealii::ExcMessage(
          "The current implementation only supports searching for particles in cells on levels finer than or equal to the level on which particles are stored."));

      dealii::TriaIterator<dealii::CellAccessor<dim>> relevant_cell = cell;
      while (relevant_cell->level() > level_to_store_particles)
        {
          relevant_cell = relevant_cell->parent();
        }

      return relevant_cell;
    }

    boost::container::small_vector<particle_accessor, max_particles_per_active_cell>
    find_relevant_particles(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) const
    {
      // TODO: This function currently assumes that the neighbor cells are on the same level
      Assert(cell->level() == level_to_store_particles,
             dealii::ExcMessage(
               "You must provide a cell id of a cell on the level on which particles are stored."));

      boost::container::small_vector<particle_accessor, max_particles_per_active_cell>
        relevant_particles;

      for (const dealii::TriaIterator<dealii::CellAccessor<dim>> &current_cell :
           level_cell_cache.get_neighboring_cells(cell))
        {
          for (dealii::Particles::ParticleIterator<dim> &particle :
               cell_to_locally_owned_particle_cache[current_cell->index()])
            relevant_particles.emplace_back(*particle, false);

          for (const typename dealii::Particles::PropertyPool<dim>::Handle particle_handle :
               cell_to_ghost_particle_cache[current_cell->index()])
            relevant_particles.emplace_back(*properties_global_obstacles, particle_handle, true);
        }

      return relevant_particles;
    }

    template <typename ObstacleContainer>
    void
    find_relevant_particles(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell,
                            ObstacleContainer &relevant_particles) const
    {
      // TODO: This function currently assumes that the neighbor cells are on the same level
      Assert(cell->level() == level_to_store_particles,
             dealii::ExcMessage(
               "You must provide a cell id of a cell on the level on which particles are stored."));

      for (const dealii::TriaIterator<dealii::CellAccessor<dim>> &current_cell :
           level_cell_cache.get_neighboring_cells(cell))
        {
          for (dealii::Particles::ParticleIterator<dim> &particle :
               cell_to_locally_owned_particle_cache[current_cell->index()])
            relevant_particles.emplace_back(*particle, false);

          for (const typename dealii::Particles::PropertyPool<dim>::Handle particle_handle :
               cell_to_ghost_particle_cache[current_cell->index()])
            relevant_particles.emplace_back(*properties_global_obstacles, particle_handle, true);
        }
    }

    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    contact_particles(const DEMParticleAccessor<dim, number> &particle,
                      const number                            relative_tolerance) const;

    void
    prepare_for_serialization()
    {
      obstacle_handler->prepare_for_serialization();
    }

    void
    deserialize()
    {
      obstacle_handler->deserialize();
    }

    unsigned int
    n_global_particles() const
    {
      return obstacle_handler->n_global_particles();
    }

    unsigned int
    n_locally_owned_particles() const
    {
      return obstacle_handler->n_locally_owned_particles();
    }

    std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
    locally_owned_particle_range() const
    {
      return std::ranges::subrange<ParticleIterator<dim, number>>(
        ParticleIterator<dim, number>(obstacle_handler->begin()),
        ParticleIterator<dim, number>(obstacle_handler->end()));
    }

    std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
    ghost_particle_range() const
    {
      return std::ranges::subrange<ParticleIterator<dim, number>>(
        ParticleIterator<dim, number>(*properties_global_obstacles, 0),
        ParticleIterator<dim, number>(*properties_global_obstacles,
                                      properties_global_obstacles->n_registered_slots()));
    }

    void
    insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                            const std::vector<std::vector<number>>        &obstacle_properties)
    {
      Assert(obstacle_locations.size() == obstacle_properties.size(),
             dealii::ExcMessage(
               "The number of obstacle locations and obstacle properties must be the same."));

      // Update the maximum particle radius based on the received particles. As this function is a
      // collective operation, we can be sure that all processes receive the same particles and thus
      // have the same maximum radius after this step. There is no need for an additional global
      // reduction operation.
      for (const auto &properties : obstacle_properties)
        {
          max_particle_radius =
            std::max(max_particle_radius, properties[ObstacleType::Properties::radius]);
        }

      std::vector<dealii::BoundingBox<dim>> local_bounding_box =
        dealii::GridTools::compute_mesh_predicate_bounding_box(
          obstacle_handler->get_triangulation(), dealii::IteratorFilters::LocallyOwnedCell());
      std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_box =
        dealii::Utilities::MPI::all_gather(mpi_communicator, local_bounding_box);

      obstacle_handler->insert_global_particles(
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ?
          obstacle_locations :
          std::vector<dealii::Point<dim, number>>{},
        global_bounding_box,
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ?
          obstacle_properties :
          std::vector<std::vector<number>>{});
    }

    std::size_t
    properties_serialized_size_in_bytes(unsigned int n_properties) const
    {
      // Location + properties
      return dim * sizeof(double) + n_properties * sizeof(double);
    }

    void *
    write_particle_properties_to_memory(
      void                                           *data_pointer,
      const dealii::Particles::ParticleIterator<dim> &particle) const
    {
      double *pdata = reinterpret_cast<double *>(data_pointer);

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

    void
    read_particle_properties_from_memory(
      void                                                       *read_data_pointer,
      dealii::Particles::PropertyPool<dim>                       &write_property_pool,
      const typename dealii::Particles::PropertyPool<dim>::Handle write_handle,
      const unsigned                                              n_properties)
    {
      const double *pdata = reinterpret_cast<const double *>(read_data_pointer);

      dealii::Point<dim> location;
      for (unsigned int i = 0; i < dim; ++i)
        location[i] = *pdata++;
      write_property_pool.set_location(write_handle, location);

      if (n_properties > 0)
        {
          const dealii::ArrayView<double> particle_properties =
            write_property_pool.get_properties(write_handle);
          for (unsigned int i = 0; i < n_properties; ++i)
            particle_properties[i] = *pdata++;
        }
    }

    void
    update_ghost_particle_properties()
    {
      std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
      for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++rank)
        {
          if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator) and
              not ghost_particle_update_cache.particles_to_send[rank].empty())
            {
              std::vector<char> send_buffer(
                ghost_particle_update_cache.particles_to_send[rank].size() *
                properties_serialized_size_in_bytes(ObstacleType::n_obstacle_properties));

              unsigned iter = 0;
              for (dealii::Particles::ParticleIterator<dim> particle :
                   ghost_particle_update_cache.particles_to_send[rank])
                {
                  write_particle_properties_to_memory(send_buffer.data() +
                                                        iter *
                                                          properties_serialized_size_in_bytes(
                                                            ObstacleType::n_obstacle_properties),
                                                      particle);
                  iter++;
                }

              send_futures.push_back(
                dealii::Utilities::MPI::isend(send_buffer, mpi_communicator, rank, 1));
            }
        }

      std::vector<dealii::Utilities::MPI::Future<std::vector<char>>> recv_futures;
      recv_futures.reserve(ghost_particle_update_cache.n_particles_to_receive.size());
      for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++rank)
        {
          if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator) and
              ghost_particle_update_cache.n_particles_to_receive[rank] > 0)
            {
              recv_futures.push_back(dealii::Utilities::MPI::irecv<std::vector<char>>(
                obstacle_handler->get_triangulation().get_mpi_communicator(), rank, 1));
            }
        }

      unsigned j = 0;
      for (unsigned int i = 0; i < ghost_particle_update_cache.n_particles_to_receive.size(); ++i)
        {
          if (i != dealii::Utilities::MPI::this_mpi_process(mpi_communicator) and
              ghost_particle_update_cache.n_particles_to_receive[i] > 0)
            {
              recv_futures[j].wait();
              // Note: recv_futures and n_particles_to_receive have the same order, so we can use
              // the same index to access both
              std::vector<char> recv_buffer = recv_futures[j].get();
              for (unsigned int p = 0; p < ghost_particle_update_cache.n_particles_to_receive[i];
                   ++p)
                {
                  read_particle_properties_from_memory(
                    recv_buffer.data() +
                      p * properties_serialized_size_in_bytes(ObstacleType::n_obstacle_properties),
                    *properties_global_obstacles,
                    ghost_particle_update_cache.rank_ghost_particle_start_handle[i] + p,
                    ObstacleType::n_obstacle_properties);
                }
              ++j;
            }
        }

      for (auto &future : send_futures)
        {
          future.wait();
        }
    }

    struct GhostParticleUpdateCache
    {
      void
      reset(MPI_Comm mpi_communicator)
      {
        particles_to_send.clear();
        particles_to_send.resize(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator));
        n_particles_to_receive.clear();
        n_particles_to_receive.resize(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator), 0);
        rank_ghost_particle_start_handle.clear();
        rank_ghost_particle_start_handle.resize(
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator), 0);
      }

      // index is the rank to whome to send the corresponding particles. Must be sorted with respect
      // to global particle id.
      std::vector<std::vector<dealii::Particles::ParticleIterator<dim>>> particles_to_send;

      // index is the rank from which to receive the corresponding particles
      std::vector<unsigned int> n_particles_to_receive;

      std::vector<unsigned int> rank_ghost_particle_start_handle;
    } ghost_particle_update_cache;

    void
    sort_particles_into_subdomains_and_cells()
    {
      sort_particles_into_local_level_cells();
      communicate_ghost_particles();

      for (dealii::Particles::ParticleIterator<dim> particle = obstacle_handler->begin();
           particle != obstacle_handler->end();
           ++particle)
        particle_id_to_iterator_cache[particle->get_id()] = particle;
    }

    void
    register_particle_output(Postprocessor<dim, number> &postprocessor)
    {
      const auto [property_names, property_component_interpretations] =
        ObstacleType::get_property_names_and_component_interpretation();

      postprocessor.register_obstacle_output(obstacle_handler.get(),
                                             property_names,
                                             property_component_interpretations);
    }

    void
    prepare_for_coarsening_and_refinement()
    {
      obstacle_handler->prepare_for_coarsening_and_refinement();
    }

    void
    unpack_after_coarsening_and_refinement()
    {
      obstacle_handler->unpack_after_coarsening_and_refinement();
      level_cell_partitioner.build_pattern(level_to_store_particles);
      sort_particles_into_subdomains_and_cells();
    }

    std::vector<std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
    get_cell_to_ghost_particle_cache() const
    {
      return cell_to_ghost_particle_cache;
    }

    unsigned int
    n_locally_relevant_particles()
    {
      unsigned int n_locally_relevant_particles = n_locally_owned_particles();
      for (const auto &particles : cell_to_ghost_particle_cache)
        {
          n_locally_relevant_particles += particles.size();
        }
      return n_locally_relevant_particles;

      for (const auto &particles : cell_to_locally_owned_particle_cache)
        {
          n_locally_relevant_particles += particles.size();
        }
      return n_locally_relevant_particles;
    }

    unsigned int
    n_ghost_particles() const
    {
      unsigned int n_ghost_particles = 0;
      for (const auto &particles : cell_to_ghost_particle_cache)
        {
          n_ghost_particles += particles.size();
        }
      return n_ghost_particles;
    }

    void
    compress();

    MPI_Comm
    get_mpi_communicator() const
    {
      return mpi_communicator;
    }


  private:
    /// Handler managing the locally owned obstacles in the domain.
    std::unique_ptr<dealii::Particles::ParticleHandler<dim>> obstacle_handler;

    /// Property pool containing the properties of all global obstacles, stored locally on each
    /// MPI rank.
    mutable std::unique_ptr<dealii::Particles::PropertyPool<dim>> properties_global_obstacles;

    /// A map that associates each cell on the specified level to store particles on with the
    /// particle iterators of the locally owned particles that are located in that cell. This cache
    /// is used to efficiently retrieve locally owned particles for a given cell without having to
    /// search through all particles.
    /// The key is the global level cell index
    mutable std::vector<std::vector<dealii::Particles::ParticleIterator<dim>>>
      cell_to_locally_owned_particle_cache;

    /// Same as above but for ghost particles. As ghost particles are stored in a separate property
    /// pool, we need to store their handles instead of their iterators. Note that the cells a ghost
    /// particle is associated with might be on a lower (coarser) than the one the particles should
    /// be stored on, as it is not guaranteed that the cell on the specified level in which the
    /// ghost particle lives is available on the current rank.
    mutable std::vector<std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
      cell_to_ghost_particle_cache;

    /// A map that associates each MPI rank with the handles in the property pool of the ghost
    /// particles storing those particles which are owned by the corresponding rank.
    mutable std::vector<std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
      rank_to_handle;

    /// A map that associates each neighbor rank with the currently stored number of ghost particles
    mutable std::vector<unsigned int> rank_to_n_ghost_particles;

    // Cache mapping particle ids to their corresponding particle iterators in the property pool.
    // This is used to efficiently compress particle properties
    std::unordered_map<dealii::types::particle_index, dealii::Particles::ParticleIterator<dim>>
      particle_id_to_iterator_cache;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    /// Timer data for profiling the obstacle search operations.
    dealii::TimerOutput &timer;

    /// The level of the triangulation at which particles are stored.
    int level_to_store_particles = 0;

    /// Variable to help keeping track of the maximum radius of the particles in the domain, which
    /// is relevant for determining the level at which to store particles.
    number max_particle_radius = 0;

    /**
     * @brief Deregisters all particles from the global obstacle property pool.
     */
    void
    deregister_property_pool() const;

    void
    sort_particles_into_local_level_cells();

    void
    communicate_ghost_particles();

    // similar to dealii::Particles::Particle functionality but also sends cell information on level
    // of interest as well as returns the handler in the new property pool when storing the
    // received particles.
    void *
    write_particle_data_to_memory(
      void                                                     *data_pointer,
      const dealii::Particles::ParticleIterator<dim>            particle,
      const typename dealii::Triangulation<dim>::cell_iterator &cell) const;

    struct ReceivedParticleData
    {
      typename dealii::Particles::PropertyPool<dim>::Handle handle;
      dealii::CellId                                        cell_id;
    };


    ReceivedParticleData
    read_particle_data_from_memory(void                                 *data_pointer,
                                   dealii::Particles::PropertyPool<dim> &property_pool,
                                   const unsigned                        n_properties) const;

    std::size_t
    serialized_size_in_bytes(unsigned int n_properties) const;

    dealii::Triangulation<dim> const *triangulation;

    LevelCellCommunicationPattern<dim> level_cell_partitioner;

    LevelCellCache<dim> level_cell_cache;
  };

  template <int dim, typename number>
  class DynamicUpdateController
  {
  public:
    /**
     * Controller for dynamic updates of the obstacle data structure.
     *
     * @param obstacle_data_structure The obstacle data structure to be monitored for updates.
     * @param skin_thickness The thickness of the "skin" around the obstacles  that triggers
     * the update when exceeded by the sum of the two largest particle displacements.
     */
    DynamicUpdateController(const CellListParticleHandler<dim, number> &obstacle_data_structure,
                            const number                                skin_thickness)
      : obstacle_data_structure(obstacle_data_structure)
      , skin_thickness(skin_thickness)
    {}


    bool
    update_required() const
    {
      Assert(
        previous_obstacle_locations.size() == obstacle_data_structure.n_global_particles(),
        dealii::ExcMessage(
          "The size of the previous obstacle location cache must be equal to the number of globally "
          "owned particles in the obstacle data structure but isn't. A common cause for this error "
          "is that the function reinit_after_update() was not called after the last update of the "
          "obstacle data structure."));

      // First is largest displacement, second is second largest displacement.
      std::pair<number, number> max_particle_displacement = std::make_pair(0, 0);
      for (const auto &particle : obstacle_data_structure.locally_owned_particle_range())
        {
          const dealii::Point<dim, number> current_location = particle.get_location();
          const dealii::Point<dim, number> previous_location =
            previous_obstacle_locations[particle.id()];
          const number displacement = current_location.distance(previous_location);
          if (displacement > max_particle_displacement.first)
            {
              max_particle_displacement.first = displacement;
            }
          else if (displacement > max_particle_displacement.second)
            {
              max_particle_displacement.second = displacement;
            }
        }

      // Communicate largest displacements to all ranks
      const number max_global_displacement =
        dealii::Utilities::MPI::max(max_particle_displacement.first,
                                    obstacle_data_structure.get_mpi_communicator());

      // Locally decide whether an update is required
      bool update = false;
      if (max_global_displacement == max_particle_displacement.first)
        {
          update = (max_global_displacement + max_particle_displacement.second) > skin_thickness;
        }
      else
        {
          update = (max_global_displacement + max_particle_displacement.first) > skin_thickness;
        }

      // Communicate the local decisions and decide globally whether an update is required
      const bool global_update_required =
        dealii::Utilities::MPI::logical_or(update, obstacle_data_structure.get_mpi_communicator());

      return global_update_required;
    }

    void
    reinit_after_update()
    {
      auto create_invalid_point = []() {
        dealii::Point<dim, number> invalid_point;
        for (unsigned int d = 0; d < dim; ++d)
          invalid_point[d] = std::numeric_limits<number>::max();
        return invalid_point;
      };
      previous_obstacle_locations.clear();
      // TODO This can be an out of bounds access when particles leave the domain
      previous_obstacle_locations.resize(obstacle_data_structure.n_global_particles(),
                                         create_invalid_point());
      for (const auto &particle : obstacle_data_structure.locally_owned_particle_range())
        previous_obstacle_locations[particle.id()] = particle.get_location();
    }

  private:
    std::vector<dealii::Point<dim, number>> previous_obstacle_locations;

    const CellListParticleHandler<dim, number> &obstacle_data_structure;

    number skin_thickness;
  };
} // namespace MeltPoolDG
