#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/particles/obstacle_field.hpp>

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
  template <int dim, typename VectorizedArrayType, typename ObstacleType>
  VectorizedArrayType
  discontinuous_mask_function(const dealii::Point<dim, VectorizedArrayType> &location,
                              dealii::Particles::PropertyPool<dim>          &property_pool,
                              const typename dealii::Particles::PropertyPool<dim>::Handle handle)
  {
    dealii::Point<dim, VectorizedArrayType> vectorized_obstacle_location;
    for (auto i = 0; i < dim; ++i)
      vectorized_obstacle_location[i] = VectorizedArrayType(property_pool.get_location(handle)[i]);
    const VectorizedArrayType distance = location.distance(vectorized_obstacle_location);
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
      distance,
      VectorizedArrayType(
        ObstacleType::get_property(property_pool, handle, ObstacleType::Properties::radius)),
      VectorizedArrayType(1.),
      VectorizedArrayType(0.));
  }

  template <int dim, typename VectorizedArrayType, typename ObstacleType>
  VectorizedArrayType
  mask_function(const MaskFunctionType                                      mask_function_type,
                const dealii::Point<dim, VectorizedArrayType>              &location,
                dealii::Particles::PropertyPool<dim>                       &property_pool,
                const typename dealii::Particles::PropertyPool<dim>::Handle handle)
  {
    switch (mask_function_type)
      {
        case MaskFunctionType::discontinuous:
          return discontinuous_mask_function<dim, VectorizedArrayType, ObstacleType>(location,
                                                                                     property_pool,
                                                                                     handle);
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
      : relevant_obstacles(ObstacleType::n_obstacle_properties)
      , obstacle_handler(obstacle_handler)
    {}

    ~CellObstacleCache()
    {
      for (auto handle : relevant_obstacle_handles)
        relevant_obstacles.deregister_particle(handle);
    }

    /**
     * Updates the cache. Checks whether the provided cell iterators differ from the
     * previously cached ones, and if so, refreshes the stored obstacle data accordingly.
     *
     * @param cells Array of cell iterators for which the cache should be updated.
     */
    void
    update_cache(const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells)
    {
      // check if cache is still valid
      if (cells.size() == cached_cells.size() &&
          std::equal(cells.begin(), cells.end(), cached_cells.begin()))
        return;

      for (auto handle : relevant_obstacle_handles)
        relevant_obstacles.deregister_particle(handle);

      relevant_obstacle_handles.clear();

      relevant_obstacle_handles = obstacle_handler.get_obstacles_in_cell(relevant_obstacles, cells);
      cached_cells              = cells;
    }

    /// Property pool used to cache obstacle properties relevant for the current cell batch.
    dealii::Particles::PropertyPool<dim> relevant_obstacles;

    /// Handles of the particles which properties are stored in the @p relevant_obstacles
    /// property pool.
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> relevant_obstacle_handles;

    /// Reference to the obstacle handler managing all obstacles in the domain.
    const ObstacleField<dim, number, ObstacleType> &obstacle_handler;

    /// Cells, for which the object cache is currently valid.
    std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cached_cells;
  };
} // namespace MeltPoolDG
