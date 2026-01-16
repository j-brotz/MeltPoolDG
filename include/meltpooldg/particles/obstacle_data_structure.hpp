#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/property_pool.h>

#include <memory>
#include <vector>

namespace MeltPoolDG
{
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
    explicit ObstacleDataStructure(ObstacleDataStructureType &&obstacle_data_structure);

    /**
     * @brief Reinitializes the internal data structure.
     */
    void
    reinit();

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
                          const dealii::CellAccessor<dim>      &cell) const;

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
    get_obstacles_in_cell_batch(dealii::Particles::PropertyPool<dim>  &dst,
                                const dealii::MatrixFree<dim, number> &matrix_free,
                                const unsigned int                     cell_batch_id) const;

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
      get_obstacles_in_cell_batch(dealii::Particles::PropertyPool<dim>  &dst,
                                  const dealii::MatrixFree<dim, number> &matrix_free,
                                  const unsigned int                     cell_batch_id) const = 0;
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
      explicit ObstacleDataStructureModel(ObstacleDataStructureType &&obstacle_data_structure);

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      void
      reinit() override;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                            const dealii::CellAccessor<dim>      &cell) const override;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell_batch(dealii::Particles::PropertyPool<dim>  &dst,
                                  const dealii::MatrixFree<dim, number> &matrix_free,
                                  const unsigned int cell_batch_id) const override;

    private:
      const ObstacleDataStructureType obstacle_data_structure;
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
    ObstacleCompleteDomainSearch(const dealii::Triangulation<dim> &triangulation,
                                 const dealii::Mapping<dim>       &mapping);

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

    dealii::Particles::ParticleIterator<dim>
    begin() const
    {
      return obstacle_handler.begin();
    }

    dealii::Particles::ParticleIterator<dim>
    end() const
    {
      return obstacle_handler.end();
    }

    dealii::Particles::ParticleIterator<dim>
    begin()
    {
      return obstacle_handler.begin();
    }

    dealii::Particles::ParticleIterator<dim>
    end()
    {
      return obstacle_handler.end();
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
     *
     * @note While this function technically does the same as @p get_obstacles_in_cell_batch(), it
     * is significantly slower. Therefore whenever possible, prefer using the batch version.
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
    get_obstacles_in_cell_batch(dealii::Particles::PropertyPool<dim>  &dst,
                                const dealii::MatrixFree<dim, number> &matrix_free,
                                const unsigned int                     cell_batch_id) const;

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

    /**
     * @brief Serializes the internal state of the class.
     *
     * This function forwards the call to dealii::ParticleHandler::serialize(). See the
     * documentation of that function for further details.
     *
     * @param ar       The archive used for serialization or deserialization.
     * @param version  The serialization version.
     */
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int version)
    {
      obstacle_handler.serialize(ar, version);
    }

    /**
     * @brief Prepares this object for serialization.
     *
     * This function forwards the call to dealii::ParticleHandler::prepare_for_serialization(). See
     * the documentation of that function for further details.
     */
    void
    prepare_for_serialization()
    {
      obstacle_handler.prepare_for_serialization();
    }

    /**
     * @brief Performs the objects deserialization.
     *
     * This function forwards the call to dealii::ParticleHandler::deserialize(). See the
     * documentation of that function for further details.
     */
    void
    deserialize()
    {
      // Assumes that triangulation.load() has already been called!
      obstacle_handler.deserialize();
    }

    /**
     * @brief Return a reference to the underlying particle handler of this object.
     *
     * @return The used particle handler of the current object.
     */
    dealii::Particles::ParticleHandler<dim> &
    get_particle_handler()
    {
      return obstacle_handler;
    }

    const dealii::Particles::ParticleHandler<dim> &
    get_particle_handler() const
    {
      return obstacle_handler;
    }

    void
    insert_obstacles(const dealii::Triangulation<dim>              &triangulation,
                     const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                     const std::vector<std::vector<number>>        &obstacle_properties)
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
    }

  private:
    /// Handler responsible for managing obstacle particles within the computational domain.
    dealii::Particles::ParticleHandler<dim> obstacle_handler;

    /// Property pool containing the properties of all global obstacles, stored locally on each
    /// MPI rank.
    mutable dealii::Particles::PropertyPool<dim> properties_global_obstacles;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    /**
     * @brief Deregisters all particles from the global obstacle property pool.
     */
    void
    deregister_property_pool() const;

    /**
     * Checks if the particle with the given handle is relevant for the specified cell. If yes, it will be added to @p properties_global_obstacles.

     * @return True if the particle is in the cell, false otherwise.
     */
    bool
    process_particle_in_cell(
      const typename dealii::Particles::PropertyPool<dim>::Handle        &src_handle,
      const dealii::CellAccessor<dim>                                    &cell,
      dealii::Particles::PropertyPool<dim>                               &dst,
      std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> &target_handles) const;
  };
} // namespace MeltPoolDG