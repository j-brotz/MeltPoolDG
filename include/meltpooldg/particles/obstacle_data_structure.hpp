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

#include "meltpooldg/particles/particle_accessor.hpp"
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/particle_iterator.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>

#include <boost/container/small_vector.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <ranges>
#include <vector>

#include "mpi.h"

namespace MeltPoolDG
{
  enum class ObstacleDataStructureType
  {
    CompleteDomainSearch,
    CellBasedSearch
  };



  /**
   * @brief Interface class for obstacle data structures supporting efficient spatial queries,
   * such as nearest-neighbor searches.
   *
   * This interface abstracts different obstacle search algorithms using type erasure, allowing
   * flexible integration of various implementations. Any obstacle search algorithm must conform
   * to this interface to be used within the framework.
   *
   * The following member functions must be implemented:
   * - @p reinit(): Reinitializes the internal state. This should be called after obstacle positions
   *   have changed or the underlying triangulation has been modified.
   * - @p get_obstacles_in_cell(): Identifies all obstacles that partially or fully occupy a given
   *   cell and stores their properties in a specified destination.
   * - @p get_obstacles_in_cell_batch(): Identifies all obstacles that intersect with any cell in a
   *   given cell batch and stores their properties in a specified destination.
   *
   * See below for more details.
   */
  template <int dim, typename number>
  struct ObstacleDataStructure
  {
  public:
    /**
     * Constructor. Store the passed obstacle data structure internally.
     *
     * @param obstacle_data_structure Concrete obstacle data structure used in the class.
     */
    template <typename ObstacleDataStructureType>
    explicit ObstacleDataStructure(ObstacleDataStructureType &&obstacle_data_structure_in)
      : obstacle_data_structure_pimpl(
          std::make_unique<ObstacleDataStructureModel<ObstacleDataStructureType>>(
            std::move(obstacle_data_structure_in)))
    {}

    /**
     * @brief Reinitializes the internal data structure.
     */
    void
    reinit()
    {
      obstacle_data_structure_pimpl->reinit();
    }

    /**
     * @brief Identify obstacles that partially or fully occupy the specified cell, and store their
     * properties in the destination property pool.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param cell The cell of interest.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    boost::container::small_vector<DEMParticleAccessor<dim, number>, 12>
    get_obstacles_in_cell(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) const
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell(cell);
    }


    /**
     * @brief Identify obstacles that partially or fully occupy any cell in the specified cell batch,
     * and store their properties in the destination property pool.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param matrix_free MatrixFree object associated with the current cell batch.
     * @param cell_batch_id Index of the cell batch to be examined.
     * @param n_lanes Number of vectorization lanes in the cell batch (i.e., the number of cells in
     * the batch).
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<MeltPoolDG::DEMParticleAccessor<dim, number>>
    get_obstacles_in_cell(
      const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell(cells);
    }

    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    contact_particles(const DEMParticleAccessor<dim, number> &particle,
                      const number                            relative_tolerance) const
    {
      return obstacle_data_structure_pimpl->contact_particles(particle, relative_tolerance);
    }

    void
    compress()
    {
      obstacle_data_structure_pimpl->compress();
    }

    void
    prepare_for_serialization()
    {
      obstacle_data_structure_pimpl->prepare_for_serialization();
    }

    void
    deserialize()
    {
      obstacle_data_structure_pimpl->deserialize();
    }

    unsigned int
    n_global_particles() const
    {
      return obstacle_data_structure_pimpl->n_global_particles();
    }

    unsigned int
    n_locally_owned_particles() const
    {
      return obstacle_data_structure_pimpl->n_locally_owned_particles();
    }

    unsigned int
    n_ghost_particles() const
    {
      return obstacle_data_structure_pimpl->n_ghost_particles();
    }

    std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
    locally_owned_particle_range()
    {
      return obstacle_data_structure_pimpl->locally_owned_particle_range();
    }

    std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
    ghost_particle_range()
    {
      return obstacle_data_structure_pimpl->ghost_particle_range();
    }

    void
    insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                            const std::vector<std::vector<number>>        &obstacle_properties)
    {
      obstacle_data_structure_pimpl->insert_global_particles(obstacle_locations,
                                                             obstacle_properties);
    }

    void
    update_ghost_particle_properties()
    {
      obstacle_data_structure_pimpl->update_ghost_particle_properties();
    }

    void
    sort_particles_into_subdomains_and_cells()
    {
      obstacle_data_structure_pimpl->sort_particles_into_subdomains_and_cells();
    }

    void
    register_particle_output(Postprocessor<dim, number> &postprocessor)
    {
      obstacle_data_structure_pimpl->register_particle_output(postprocessor);
    }

    void
    prepare_for_coarsening_and_refinement()
    {
      obstacle_data_structure_pimpl->prepare_for_coarsening_and_refinement();
    }

    void
    unpack_after_coarsening_and_refinement()
    {
      obstacle_data_structure_pimpl->unpack_after_coarsening_and_refinement();
    }

  private:
    /**
     * @brief Concept: Abstract interface for obstacle data structures.
     *
     * Implementations must define how to reinitialize the structure, and  how to extract obstacles
     * from single cells or batches.See public interface for detailed function descriptions.
     */
    struct ObstacleDataStructureConcept
    {
      virtual ~ObstacleDataStructureConcept() = default;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      virtual void
      reinit() = 0;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      virtual boost::container::small_vector<DEMParticleAccessor<dim, number>, 12>
      get_obstacles_in_cell(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) const = 0;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      virtual std::vector<MeltPoolDG::DEMParticleAccessor<dim, number>>
      get_obstacles_in_cell(
        const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const = 0;

      virtual boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
      contact_particles(const DEMParticleAccessor<dim, number> &particle,
                        const number                            relative_tolerance) const = 0;

      virtual void
      prepare_for_serialization() = 0;

      virtual void
      deserialize() = 0;

      virtual void
      compress() = 0;

      virtual unsigned int
      n_global_particles() const = 0;

      virtual unsigned int
      n_locally_owned_particles() const = 0;

      virtual std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
      locally_owned_particle_range() = 0;

      virtual std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
      ghost_particle_range() = 0;

      virtual void
      insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                              const std::vector<std::vector<number>> &obstacle_properties) = 0;

      virtual unsigned int
      n_ghost_particles() const = 0;

      virtual void
      update_ghost_particle_properties() = 0;

      virtual void
      sort_particles_into_subdomains_and_cells() = 0;

      virtual void
      register_particle_output(Postprocessor<dim, number> &postprocessor) = 0;

      virtual void
      prepare_for_coarsening_and_refinement() = 0;

      virtual void
      unpack_after_coarsening_and_refinement() = 0;
    };

    /**
     * @brief Model required for type erasure. See public interface for detailed function
     * descriptions.
     */
    template <typename ObstacleDataStructureType>
    struct ObstacleDataStructureModel final : public ObstacleDataStructureConcept
    {
      /**
       * Constructor. Stores the given obstacle data structure internally.
       *
       * @param obstacle_data_structure Data structure of the obstacles.
       */
      explicit ObstacleDataStructureModel(ObstacleDataStructureType &&obstacle_data_structure)
        : obstacle_data_structure(std::move(obstacle_data_structure))
      {}

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      void
      reinit() override
      {
        obstacle_data_structure.reinit();
      }

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      boost::container::small_vector<DEMParticleAccessor<dim, number>, 12>
      get_obstacles_in_cell(
        const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) const override
      {
        return obstacle_data_structure.get_obstacles_in_cell(cell);
      }

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      std::vector<MeltPoolDG::DEMParticleAccessor<dim, number>>
      get_obstacles_in_cell(
        const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const override
      {
        return obstacle_data_structure.get_obstacles_in_cell(cells);
      }

      void
      compress() override
      {
        obstacle_data_structure.compress();
      }

      boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
      contact_particles(const DEMParticleAccessor<dim, number> &particle,
                        const number                            relative_tolerance) const override
      {
        return obstacle_data_structure.contact_particles(particle, relative_tolerance);
      }

      void
      prepare_for_serialization() override
      {
        obstacle_data_structure.prepare_for_serialization();
      }

      void
      deserialize() override
      {
        obstacle_data_structure.deserialize();
      }

      unsigned int
      n_global_particles() const override
      {
        return obstacle_data_structure.n_global_particles();
      }

      unsigned int
      n_locally_owned_particles() const override
      {
        return obstacle_data_structure.n_locally_owned_particles();
      }

      unsigned int
      n_ghost_particles() const override
      {
        return obstacle_data_structure.n_ghost_particles();
      }

      std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
      locally_owned_particle_range() override
      {
        return obstacle_data_structure.locally_owned_particle_range();
      }

      std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
      ghost_particle_range() override
      {
        return obstacle_data_structure.ghost_particle_range();
      }

      void
      insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                              const std::vector<std::vector<number>> &obstacle_properties) override
      {
        obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
      }

      void
      update_ghost_particle_properties() override
      {
        obstacle_data_structure.update_ghost_particle_properties();
      }

      void
      sort_particles_into_subdomains_and_cells() override
      {
        obstacle_data_structure.sort_particles_into_subdomains_and_cells();
      }

      void
      register_particle_output(Postprocessor<dim, number> &postprocessor) override
      {
        obstacle_data_structure.register_particle_output(postprocessor);
      }

      void
      prepare_for_coarsening_and_refinement() override
      {
        obstacle_data_structure.prepare_for_coarsening_and_refinement();
      }

      void
      unpack_after_coarsening_and_refinement() override
      {
        obstacle_data_structure.unpack_after_coarsening_and_refinement();
      }


    private:
      ObstacleDataStructureType obstacle_data_structure;
    };

    /// Pointer to the concrete obstacle data structure used within this class.
    std::unique_ptr<ObstacleDataStructureConcept> obstacle_data_structure_pimpl;
  };

  /**
   * @brief A simple search utility that performs a linear scan over all particles.
   *
   * This struct implements a very basic search algorithm by iterating over all particles
   * in the domain and individually checking whether each one satisfies the desired condition.
   *
   * @note This approach is not optimized for performance and is intended primarily for
   * debugging, prototyping, or use in small-scale simulations.
   */
  template <int dim, typename number, typename ObstacleType>
  struct ObstacleCompleteDomainSearch
  {
  public:
    static constexpr std::size_t max_particles_per_active_cell = 20;
    using particle_accessor                                    = DEMParticleAccessor<dim, number>;

    ObstacleCompleteDomainSearch(const dealii::Triangulation<dim>      &triangulation,
                                 const dealii::Mapping<dim>            &mapping,
                                 dealii::TimerOutput                   &timer,
                                 const dealii::MatrixFree<dim, number> *matrix_free = nullptr);

    /**
     * @brief Destructor. Explicitly deregisters all particles from the global obstacle property
     * pool.
     */
    ~ObstacleCompleteDomainSearch();

    ObstacleCompleteDomainSearch(ObstacleCompleteDomainSearch &&) = default;

    ObstacleCompleteDomainSearch &
    operator=(ObstacleCompleteDomainSearch &&) = default;

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
    reinit();

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
    void
    get_obstacles_in_cell(
      const boost::container::small_vector_base<dealii::TriaIterator<dealii::CellAccessor<dim>>>
                                                                            &cells,
      boost::container::small_vector_base<DEMParticleAccessor<dim, number>> &particles) const
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

    std::vector<MeltPoolDG::DEMParticleAccessor<dim, number>> &
    get_obstacles_in_cell_batch(const unsigned int cell_batch_id) const
    {
      return cell_batch_to_particle_cache[cell_batch_id];
    }

    void
    get_obstacles_in_cell(
      const dealii::TriaIterator<dealii::CellAccessor<dim>>                 &cell,
      boost::container::small_vector_base<DEMParticleAccessor<dim, number>> &particles) const
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

    void
    find_relevant_particles(
      const dealii::TriaIterator<dealii::CellAccessor<dim>>  &cell,
      boost::container::small_vector_base<particle_accessor> &relevant_particles) const
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

      reinit();
    }

    void
    update_ghost_particle_properties()
    {
      // TODO
      sort_particles_into_subdomains_and_cells();
    }

    void
    sort_particles_into_subdomains_and_cells()
    {
      sort_particles_into_local_level_cells();
      communicate_ghost_particles();
      update_matrix_free_cache();

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

    mutable std::vector<std::vector<DEMParticleAccessor<dim, number>>> cell_batch_to_particle_cache;

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

    const dealii::MatrixFree<dim, number> *matrix_free = nullptr;

    /**
     * @brief Deregisters all particles from the global obstacle property pool.
     */
    void
    deregister_property_pool() const;

    void
    sort_particles_into_local_level_cells();

    void
    communicate_ghost_particles();

    void
    update_matrix_free_cache()
    {
      if (matrix_free != nullptr)
        {
          cell_batch_to_particle_cache.clear();
          cell_batch_to_particle_cache.resize(matrix_free->n_cell_batches());
          // dummy vectos for the cell loop, not used in the loop body
          dealii::Vector<number> dummy;

          std::function<void(const dealii::MatrixFree<dim, number> &,
                             dealii::Vector<number> &,
                             const dealii::Vector<number> &,
                             const std::pair<unsigned int, unsigned int> &)>
            cell_op = [&](const dealii::MatrixFree<dim, number> &mf,
                          dealii::Vector<number> &,
                          const dealii::Vector<number> &,
                          const std::pair<unsigned int, unsigned int> &cell_range) {
              for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
                {
                  for (unsigned int lane = 0;
                       lane < matrix_free->n_active_entries_per_cell_batch(cell);
                       ++lane)
                    {
                      auto cell_iterator = mf.get_cell_iterator(cell, lane);


                      boost::container::small_vector<DEMParticleAccessor<dim, number>, 16>
                        particles;
                      find_relevant_particles(find_particle_storage_cell(cell_iterator), particles);
                      for (DEMParticleAccessor<dim, number> particle : particles)
                        {
                          auto it = std::find_if(cell_batch_to_particle_cache[cell].begin(),
                                                 cell_batch_to_particle_cache[cell].end(),
                                                 [&](const DEMParticleAccessor<dim, number> &p) {
                                                   return p.id() == particle.id();
                                                 });
                          if (it == cell_batch_to_particle_cache[cell].end())
                            {
                              cell_batch_to_particle_cache[cell].push_back(particle);
                            }
                        }
                    }
                }
            };

          matrix_free->cell_loop(cell_op, dummy, dummy);
        }
    }

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

  template <int dim, typename number, typename ObstacleType>
  ObstacleDataStructure<dim, number>
  obstacle_data_structure_factory(const ObstacleDataStructureType   data_structure_type,
                                  const dealii::Triangulation<dim> &triangulation,
                                  const dealii::Mapping<dim>       &mapping,
                                  dealii::TimerOutput              &timer)
  {
    switch (data_structure_type)
      {
        case ObstacleDataStructureType::CompleteDomainSearch:
          return ObstacleDataStructure<dim, number>(
            ObstacleCompleteDomainSearch<dim, number, ObstacleType>(triangulation, mapping, timer));
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }
  }
} // namespace MeltPoolDG
