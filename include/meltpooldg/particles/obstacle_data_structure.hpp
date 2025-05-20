#pragma once

#include <deal.II/base/array_view.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/solution_transfer.h>

#include <iterator>
#include <memory>
#include <utility>
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
  private:
    struct ObstacleDataStructureConcept
    {
      virtual ~ObstacleDataStructureConcept() = default;

      virtual void
      reinit() = 0;

      virtual void
      get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                            const dealii::CellAccessor<dim>      &cell) const = 0;

      virtual void
      get_obstacles_in_cell_batch(
        dealii::Particles::PropertyPool<dim>  &dst,
        const dealii::MatrixFree<dim, number> &matrix_free,
        const unsigned int                     cell_batch_id,
        const unsigned int n_lanes = dealii::VectorizedArray<number>::size) const = 0;
    };

    template <typename ObstacleDataStructureType>
    struct ObstacleDataStructureModel final : public ObstacleDataStructureConcept
    {
      explicit ObstacleDataStructureModel(ObstacleDataStructureType &&obstacle_data_structure)
        : obstacle_data_structure(std::move(obstacle_data_structure))
      {}

      void
      reinit() override
      {
        obstacle_data_structure.reinit();
      }

      void
      get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                            const dealii::CellAccessor<dim>      &cell) const override
      {
        return obstacle_data_structure.compute_force_on_obstacle(dst, cell);
      }

      void
      get_obstacles_in_cell_batch(
        dealii::Particles::PropertyPool<dim>  &dst,
        const dealii::MatrixFree<dim, number> &matrix_free,
        const unsigned int                     cell_batch_id,
        const unsigned int n_lanes = dealii::VectorizedArray<number>::size) const override
      {
        return obstacle_data_structure.compute_force_on_obstacle(dst,
                                                                 matrix_free,
                                                                 cell_batch_id,
                                                                 n_lanes);
      }

    private:
      const ObstacleDataStructureType obstacle_data_structure;
    };

    std::unique_ptr<ObstacleDataStructureConcept> obstacle_data_structure_pimpl;

  public:
    template <typename ObstacleDataStructureType>
    explicit ObstacleDataStructure(ObstacleDataStructureType &&obstacle_data_structure)
      : obstacle_data_structure_pimpl(
          std::make_unique<ObstacleDataStructureModel<ObstacleDataStructureType>>(
            std::move(obstacle_data_structure)))
    {}

    void
    reinit()
    {
      obstacle_data_structure_pimpl->reinit();
    }

    void
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell(dst, cell);
    }

    void
    get_obstacles_in_cell_batch(
      dealii::Particles::PropertyPool<dim>  &dst,
      const dealii::MatrixFree<dim, number> &matrix_free,
      const unsigned int                     cell_batch_id,
      const unsigned int                     n_lanes = dealii::VectorizedArray<number>::size) const
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell_batch(dst,
                                                                        matrix_free,
                                                                        cell_batch_id,
                                                                        n_lanes);
    }
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
    explicit ObstacleCompleteDomainSearch(
      const dealii::Particles::ParticleHandler<dim> &obstacle_handler);

    /**
     * @brief Reinitializes the internal data structure by synchronizing obstacle data across all
     * MPI processes.
     *
     * This function communicates all locally owned obstacles to every other process in the MPI
     * communicator. As a result, each process obtains and stores a complete local copy of all
     * obstacles, regardless of ownership. This enables fully local access to obstacle data during
     * subsequent computations.
     */
    void
    reinit() const;

    /**
     * @brief Identify obstacles that partially or fully occupy the specified cell, and store their
     * properties in the destination property pool.
     *
     * This function scans a cell and collects the properties of all obstacles that intersect with
     * the cell. These properties are then stored in the provided destination property pool.
     *
     * The identification strategy is based on a brute-force approach: the function iterates over
     * all globally known obstacles (previously synchronized), checks each one for relevance to the
     * specified cell, and includes it if applicable.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param cell The cell of interest.
     */
    void
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const;

    /**
     * @brief Identify obstacles that partially or fully occupy any cell in the specified cell batch,
     * and store their properties in the destination property pool.
     *
     * This function scans a batch of cells (represented by a MatrixFree cell batch) and collects
     * the properties of all obstacles that intersect with any cell in the batch. These properties
     * are then stored in the provided destination property pool.
     *
     * The identification strategy is based on a brute-force approach: the function iterates over
     * all globally known obstacles (previously synchronized), checks each one for relevance to the
     * specified cell batch, and includes it if applicable.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param matrix_free MatrixFree object associated with the current cell batch.
     * @param cell_batch_id Index of the cell batch to be examined.
     * @param n_lanes Number of vectorization lanes in the cell batch (i.e., the number of cells in
     * the batch).
     */
    void
    get_obstacles_in_cell_batch(
      dealii::Particles::PropertyPool<dim>  &dst,
      const dealii::MatrixFree<dim, number> &matrix_free,
      const unsigned int                     cell_batch_id,
      const unsigned int                     n_lanes = dealii::VectorizedArray<number>::size) const;

    /**
     * @brief Broadcasts obstacle properties of all locally owned particles to all MPI processes.
     *
     * This function ensures that each process has access to a complete copy of all obstacles,
     * regardless of ownership. It enables computations involving obstacles even on processes that
     * do not originally own them.
     *
     * Each process broadcasts its locally owned obstacles in turn, including both their location
     * and
     * associated properties. The data is stored in the @p properties_global_obstacles structure.
     */
    void
    broadcast_global_particles() const;

  private:
    /// Handler managing the locally owned obstacles in the domain.
    const dealii::Particles::ParticleHandler<dim> &obstacle_handler;

    /// Property pool containing the properties of all global obstacles, stored locally on each MPI
    /// rank.
    mutable dealii::Particles::PropertyPool<dim> properties_global_obstacles;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;
  };
} // namespace MeltPoolDG


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
      if (ObstacleType::in_cell(properties_global_obstacles, src_handle, cell))
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
          if (ObstacleType::in_cell(properties_global_obstacles,
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