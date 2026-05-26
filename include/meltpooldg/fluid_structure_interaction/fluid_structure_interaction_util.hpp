#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/utilities/better_enum.hpp>

#include <functional>

namespace MeltPoolDG
{
  BETTER_ENUM(MaskFunctionType, char, discontinuous);

  /**
   * Implementation of a discontinuous mask function, which returns one if a given coordinate is
   * inside an obstacle and zero otherwise.
   *
   * @param location Coordinates at which the mask function shall be computed.
   * @param property_pool Property pool holding all required information about the obstacles at
   * the given locations.
   * @param handle Handle to identify relevant obstacles in the property pool.
   *
   * @return Value of mask function at given coordinates.
   */
  template <int dim, typename number, typename VectorizedArrayType, typename ObstacleType>
  inline DEAL_II_ALWAYS_INLINE VectorizedArrayType
  discontinuous_mask_function(const dealii::Point<dim, VectorizedArrayType> &location,
                              const DEMParticleAccessor<dim, number>        &particle)
  {
    dealii::Point<dim, VectorizedArrayType> vectorized_obstacle_location;
    for (auto i = 0; i < dim; ++i)
      vectorized_obstacle_location[i] = VectorizedArrayType(particle.get_location()[i]);
    const VectorizedArrayType distance_square =
      location.distance_square(vectorized_obstacle_location);
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
      distance_square,
      VectorizedArrayType(particle.radius() * particle.radius()),
      VectorizedArrayType(1.),
      VectorizedArrayType(0.));
  }

  template <int dim, typename number, typename VectorizedArrayType, typename ObstacleType>
  inline DEAL_II_ALWAYS_INLINE VectorizedArrayType
  mask_function(const MaskFunctionType                         mask_function_type,
                const dealii::Point<dim, VectorizedArrayType> &location,
                const DEMParticleAccessor<dim, number>        &particle)
  {
    switch (mask_function_type)
      {
        case MaskFunctionType::discontinuous:
          return discontinuous_mask_function<dim, number, VectorizedArrayType, ObstacleType>(
            location, particle);
        default:
          AssertThrow(false, dealii::ExcMessage("Unknown mask function type."));
      }
    return VectorizedArrayType(0.);
  }


  /**
   * A cache data structure used to store particle-related data at the cell level
   * during particle–fluid coupling computations. Its primary purpose is to avoid
   * recomputing the same cell data for every quadrature point within a
   * matrix-free loop by caching the relevant information once per cell.
   */
  template <int dim, typename number, typename ObstacleType>
  struct CellObstacleCache
  {
    CellObstacleCache(const ObstacleField<dim, number, ObstacleType> &obstacle_handler)
      : obstacle_handler(obstacle_handler)
    {}

    /**
     * Updates the cache. Checks whether the provided cell iterators differ from the
     * previously cached ones, and if so, refreshes the stored obstacle data accordingly.
     *
     * @param cells Array of cell iterators for which the cache should be updated.
     */
    inline DEAL_II_ALWAYS_INLINE void
    update_cache(const unsigned int cell_batch_id)
    {
      // check if cache is still valid
      if (cell_batch_id == cached_cell_batch_id)
        return;
      obstacle_cache.clear();
      obstacle_handler.get_obstacles_in_cell_batch(cell_batch_id, obstacle_cache);
    }
    /// Handles of the particles which properties are stored in the @p relevant_obstacles
    /// property pool.
    boost::container::small_vector<
      MeltPoolDG::DEMParticleAccessor<dim, number>,
      MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::
          max_particles_per_active_cell *
        8>
      obstacle_cache;

    /// Reference to the obstacle handler managing all obstacles in the domain.
    const ObstacleField<dim, number, ObstacleType> &obstacle_handler;

    /// Cells, for which the object cache is currently valid.
    const unsigned int cached_cell_batch_id = std::numeric_limits<unsigned int>::max();
  };
} // namespace MeltPoolDG
