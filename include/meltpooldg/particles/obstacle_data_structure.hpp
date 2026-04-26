#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/property_pool.h>

#include "meltpooldg/particles/particle_accessor.hpp"
#include <meltpooldg/particles/particle_iterator.hpp>

#include <boost/container/small_vector.hpp>

#include <cmath>
#include <memory>
#include <vector>

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
    explicit ObstacleDataStructure(ObstacleDataStructureType &&obstacle_data_structure)
      : obstacle_data_structure_pimpl(
          std::make_unique<ObstacleDataStructureModel<ObstacleDataStructureType>>(
            std::move(obstacle_data_structure)))
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
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell(dst, cell);
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
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(
      dealii::Particles::PropertyPool<dim>                               &dst,
      const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell(dst, cells);
    }

    void
    broadcast_global_particles() const
    {
      obstacle_data_structure_pimpl->broadcast_global_particles();
    }

    const dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties() const
    {
      return obstacle_data_structure_pimpl->get_global_particle_properties();
    }

    dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties()
    {
      return obstacle_data_structure_pimpl->get_global_particle_properties();
    }

    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    contact_particles(const DEMParticleAccessor<dim, number> &particle,
                      const number                            relative_tolerance) const
    {
      return obstacle_data_structure_pimpl->contact_particles(particle, relative_tolerance);
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
      virtual std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                            const dealii::CellAccessor<dim>      &cell) const = 0;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      virtual std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(
        dealii::Particles::PropertyPool<dim>                               &dst,
        const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const = 0;

      virtual void
      broadcast_global_particles() const = 0;

      virtual const dealii::Particles::PropertyPool<dim> &
      get_global_particle_properties() const = 0;

      virtual dealii::Particles::PropertyPool<dim> &
      get_global_particle_properties() = 0;

      virtual boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
      contact_particles(const DEMParticleAccessor<dim, number> &particle,
                        const number                            relative_tolerance) const = 0;
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
      std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                            const dealii::CellAccessor<dim>      &cell) const override
      {
        return obstacle_data_structure.get_obstacles_in_cell(dst, cell);
      }

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(
        dealii::Particles::PropertyPool<dim>                               &dst,
        const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const override
      {
        return obstacle_data_structure.get_obstacles_in_cell(dst, cells);
      }

      void
      broadcast_global_particles() const override
      {
        obstacle_data_structure.broadcast_global_particles();
      }

      const dealii::Particles::PropertyPool<dim> &
      get_global_particle_properties() const override
      {
        return obstacle_data_structure.get_global_particle_properties();
      }

      dealii::Particles::PropertyPool<dim> &
      get_global_particle_properties() override
      {
        return obstacle_data_structure.get_global_particle_properties();
      }

      boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
      contact_particles(const DEMParticleAccessor<dim, number> &particle,
                        const number                            relative_tolerance) const override
      {
        return obstacle_data_structure.contact_particles(particle, relative_tolerance);
      }

    private:
      ObstacleDataStructureType obstacle_data_structure;
    };

    /// Pointer to the concrete obstacle data structure used withion this class.
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
    ObstacleCompleteDomainSearch(const dealii::Particles::ParticleHandler<dim> &obstacle_handler,
                                 dealii::TimerOutput                           &timer);

    /**
     * @brief Destructor. Explicitly deregisters all particles from the global obstacle property
     * pool.
     */
    ~ObstacleCompleteDomainSearch();

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
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const;

    /**
     * @brief Identify obstacles that partially or fully occupy any cell in the specified cell batch,
     * and store their properties in the destination property pool.
     *
     * This function scans a batch of cells (represented by a MatrixFree cell batch) and
     * collects the properties of all obstacles that intersect with any cell in the batch. These
     * properties are then stored in the provided destination property pool.
     *
     * The identification strategy is based on a brute-force approach: the function iterates
     * over all globally known obstacles (previously synchronized), checks each one for
     * relevance to the specified cell batch, and includes it if applicable.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param matrix_free MatrixFree object associated with the current cell batch.
     * @param cell_batch_id Index of the cell batch to be examined.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(
      dealii::Particles::PropertyPool<dim>                               &dst,
      const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const;

    /**
     * @brief Broadcasts obstacle properties of all locally owned particles to all MPI processes.
     *
     * This function ensures that each process has access to a complete copy of all obstacles,
     * regardless of ownership. It enables computations involving obstacles even on processes
     * that do not originally own them.
     *
     * Each process broadcasts its locally owned obstacles in turn, including both their
     * location and
     * associated properties. The data is stored in the @p properties_global_obstacles structure.
     */
    void
    broadcast_global_particles() const;

    /**
     * Return a reference to a property pool containing the properties of all globally available
     * particles. The properties stored in the property pool represent those available in the field
     * when broadcast_global_particles() has been called the last time.
     */
    const dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties() const
    {
      return properties_global_obstacles;
    }

    dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties()
    {
      return properties_global_obstacles;
    }

    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    contact_particles(const DEMParticleAccessor<dim, number> &particle,
                      const number                            relative_tolerance) const;

  private:
    /// Handler managing the locally owned obstacles in the domain.
    const dealii::Particles::ParticleHandler<dim> &obstacle_handler;

    /// Property pool containing the properties of all global obstacles, stored locally on each
    /// MPI rank.
    mutable dealii::Particles::PropertyPool<dim> properties_global_obstacles;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    /// Timer data for profiling the obstacle search operations.
    dealii::TimerOutput &timer;

    /**
     * @brief Deregisters all particles from the global obstacle property pool.
     */
    void
    deregister_property_pool() const;
  };

  template <int dim, typename number, typename ObstacleType>
  struct ObstacleTriangulationDataStructure
  {
  public:
    ObstacleTriangulationDataStructure(
      const dealii::Particles::ParticleHandler<dim> &obstacle_handler,
      dealii::TimerOutput                           &timer)
      : obstacle_handler(obstacle_handler)
      , properties_global_obstacles(ObstacleType::n_obstacle_properties)
      , cell_neighbor_cache(obstacle_handler.get_triangulation())
      , timer(timer)
    {}

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
    reinit()
    {}

    static void
    setup_hyper_rectangular_triangulation(
      dealii::Triangulation<dim>                                       &triangulation,
      std::pair<dealii::Point<dim, number>, dealii::Point<dim, number>> domain_bounding_box,
      const number max_particle_influence_radius)
    {
      std::vector<unsigned int> subdivisions(dim, 1);
      for (unsigned int d = 0; d < dim; ++d)
        {
          const number domain_length = domain_bounding_box.second[d] - domain_bounding_box.first[d];
          subdivisions[d]            = std::floor(domain_length / max_particle_influence_radius);
        }

      dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        subdivisions,
                                                        domain_bounding_box.first,
                                                        domain_bounding_box.second);
    }

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
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const
    {
      return {};
    }

    /**
     * @brief Identify obstacles that partially or fully occupy any cell in the specified cell batch,
     * and store their properties in the destination property pool.
     *
     * This function scans a batch of cells (represented by a MatrixFree cell batch) and
     * collects the properties of all obstacles that intersect with any cell in the batch. These
     * properties are then stored in the provided destination property pool.
     *
     * The identification strategy is based on a brute-force approach: the function iterates
     * over all globally known obstacles (previously synchronized), checks each one for
     * relevance to the specified cell batch, and includes it if applicable.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param matrix_free MatrixFree object associated with the current cell batch.
     * @param cell_batch_id Index of the cell batch to be examined.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(
      dealii::Particles::PropertyPool<dim>                               &dst,
      const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const
    {
      return {};
    }

    /**
     * Do nothing in here
     */
    void
    broadcast_global_particles() const
    {}

    /**
     * Return a reference to a property pool containing the properties of all globally available
     * particles. The properties stored in the property pool represent those available in the field
     * when broadcast_global_particles() has been called the last time.
     */
    const dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties() const
    {
      return properties_global_obstacles;
    }

    dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties()
    {
      return properties_global_obstacles;
    }

    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    contact_particles(const DEMParticleAccessor<dim, number> &particle, const number) const
    {
      dealii::TimerOutput::Scope t(timer, "contact particle search");

      boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim> contacts;
      typename dealii::Triangulation<dim>::cell_iterator cell = particle.get_surrounding_cell();

      for (auto &other_cell :
           cell_neighbor_cache.get_neighbors_and_cell(cell->level(), cell->index()))
        {
          for (auto &other : obstacle_handler.particles_in_cell(other_cell))
            {
              DEMParticleAccessor<dim, number> other_accessor(other);
              if (other_accessor.id() != particle.id())
                {
                  contacts.emplace_back(other);
                }
            }
        }

      return contacts;
    }

  private:
    /// Handler managing the locally owned obstacles in the domain.
    const dealii::Particles::ParticleHandler<dim> &obstacle_handler;

    /// Property pool containing the properties of all global obstacles, stored locally on each
    /// MPI rank.
    mutable dealii::Particles::PropertyPool<dim> properties_global_obstacles;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    struct CellNeighborCache
    {
      CellNeighborCache(const dealii::Triangulation<dim> &triangulation)
      {
        cache_values.resize(triangulation.n_global_levels());
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
          {
            cache_values[level].resize(triangulation.n_cells(level));
          }

        for (const auto &cell : triangulation.active_cell_iterators())
          {
            const int level = cell->level();
            const int index = cell->index();
            cache_values[level][index] =
              std::set<typename dealii::Triangulation<dim>::cell_iterator>();
            for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
              {
                cache_values[level][index].insert(cell);
                if (!cell->at_boundary(f))
                  {
                    auto neighbor = cell->neighbor(f);
                    cache_values[level][index].insert(neighbor);
                    for (unsigned int n_face = 0;
                         n_face < dealii::GeometryInfo<dim>::faces_per_cell;
                         ++n_face)
                      {
                        if (!neighbor->at_boundary(n_face) and
                            neighbor->neighbor(n_face) != cell and n_face != f)
                          {
                            auto neighbor_of_neighbor = neighbor->neighbor(n_face);
                            cache_values[level][index].insert(neighbor_of_neighbor);
                            for (unsigned int nn_face = 0;
                                 nn_face < dealii::GeometryInfo<dim>::faces_per_cell;
                                 ++nn_face)
                              {
                                if (!neighbor_of_neighbor->at_boundary(nn_face) and
                                    neighbor_of_neighbor->neighbor(nn_face) != neighbor and
                                    neighbor_of_neighbor->neighbor(nn_face) != cell and
                                    nn_face != n_face and nn_face != f and nn_face < 2)
                                  {
                                    auto neighbor_of_neighbor_of_neighbor =
                                      neighbor_of_neighbor->neighbor(nn_face);
                                    cache_values[level][index].insert(
                                      neighbor_of_neighbor_of_neighbor);
                                  }
                              }
                          }
                      }
                  }
              }
          }
      }

      const std::set<typename dealii::Triangulation<dim>::cell_iterator> &
      get_neighbors_and_cell(const int level, const int index) const
      {
        return cache_values[level][index];
      }

    private:
      std::vector<std::vector<std::set<typename dealii::Triangulation<dim>::cell_iterator>>>
        cache_values;
    };

    CellNeighborCache cell_neighbor_cache;

    /// Timer data for profiling the obstacle search operations.
    dealii::TimerOutput &timer;
  };



  template <int dim, typename number, typename ObstacleType>
  ObstacleDataStructure<dim, number>
  obstacle_data_structure_factory(const ObstacleDataStructureType data_structure_type,
                                  const dealii::Particles::ParticleHandler<dim> &obstacle_handler,
                                  dealii::TimerOutput                           &timer)
  {
    switch (data_structure_type)
      {
        case ObstacleDataStructureType::CompleteDomainSearch:
          return ObstacleDataStructure<dim, number>(
            ObstacleCompleteDomainSearch<dim, number, ObstacleType>(obstacle_handler, timer));
        case ObstacleDataStructureType::CellBasedSearch:
          return ObstacleDataStructure<dim, number>(
            ObstacleTriangulationDataStructure<dim, number, ObstacleType>(obstacle_handler, timer));
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }
  }
} // namespace MeltPoolDG