#pragma once

#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <boost/container/small_vector.hpp>

#include <vector>

namespace MeltPoolDG
{
  /**
   * A cache that stores, for each cell batch in a matrix-free context, the particles relevant to
   * the cells in that batch. This is useful for efficiently accessing particle data during
   * matrix-free operations, as it avoids repeated searches for particles in each cell.
   */
  template <int dim, typename number, typename ObstacleType>
  class MatrixFreeCellBatchParticleCache
  {
  public:
    /**
     * Constructor, stores a reference to the matrix-free object internally.
     *
     * @param mf_context The matrix-free context that provides information about the cell batches
     * for which the particle cache is to be constructed.
     */
    MatrixFreeCellBatchParticleCache(const MatrixFreeContext<dim, number> mf_context)
      : mf_context(mf_context)
    {}

    /**
     * Updates the particle cache by querying the provided obstacle field for the particles
     * relevant to each cell batch in the matrix-free context. This function should be called
     * whenever the particle-cell relationship changes.
     *
     * @param particle_handler The obstacle data structure that manages the particles in the domain.
     * @param event The type of event that triggered the update.
     */
    void
    update(CellListParticleHandler<dim, number, ObstacleType> &particle_handler,
           const typename CellListParticleHandler<dim, number, ObstacleType>::NotifyEvent event);

    /**
     * Returns the particles relevant to the specified cell batch. For the definition of "relevant",
     * see the documentation of the CellListParticleHandler class.
     *
     * @param cell_batch_id The ID of the cell batch for which to retrieve particles.
     * @return A const reference to the vector of particles for the specified cell batch.
     */
    inline DEAL_II_ALWAYS_INLINE const std::vector<DEMParticleAccessor<dim, number>> &
    obstacles_in_cell_batch(const unsigned int cell_batch_id) const;

    /**
     * Same as the above const version of obstacles_in_cell_batch, but returns a non-const reference
     * to the vector of particles for the specified cell batch.
     */
    inline DEAL_II_ALWAYS_INLINE std::vector<DEMParticleAccessor<dim, number>>                              &
    obstacles_in_cell_batch(const unsigned int cell_batch_id);

  private:
    /// The matrix-free context that provides information about the cell batches for which the
    /// particle cache is constructed.
    const MatrixFreeContext<dim, number> mf_context;

    /// Vector caching, for each cell batch, the particles relevant to the cells in that batch. The
    /// index of the outer vector corresponds to the cell batch id.
    std::vector<std::vector<DEMParticleAccessor<dim, number>>> cell_batch_to_particle_cache;
  };


  template <int dim, typename number, typename ObstacleType>
  const std::vector<DEMParticleAccessor<dim, number>> &
  MatrixFreeCellBatchParticleCache<dim, number, ObstacleType>::obstacles_in_cell_batch(
    const unsigned int cell_batch_id) const
  {
    AssertIndexRange(cell_batch_id, cell_batch_to_particle_cache.size());
    return cell_batch_to_particle_cache[cell_batch_id];
  }

  template <int dim, typename number, typename ObstacleType>
  std::vector<DEMParticleAccessor<dim, number>> &
  MatrixFreeCellBatchParticleCache<dim, number, ObstacleType>::obstacles_in_cell_batch(
    const unsigned int cell_batch_id)
  {
    AssertIndexRange(cell_batch_id, cell_batch_to_particle_cache.size());
    return cell_batch_to_particle_cache[cell_batch_id];
  }
} // namespace MeltPoolDG
