#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/particles/obstacle_field.hpp>
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
  template <int dim, typename number, typename VectorizedArrayType>
  VectorizedArrayType
  discontinuous_mask_function(const dealii::Point<dim, VectorizedArrayType> &location,
                              const DEMParticleAccessor<dim, number>        &particle)
  {
    dealii::Point<dim, VectorizedArrayType> vectorized_obstacle_location;
    for (auto i = 0; i < dim; ++i)
      vectorized_obstacle_location[i] = VectorizedArrayType(particle.get_location()[i]);
    const VectorizedArrayType distance = location.distance(vectorized_obstacle_location);
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
      distance,
      VectorizedArrayType(particle.radius()),
      VectorizedArrayType(1.),
      VectorizedArrayType(0.));
  }

  template <int dim, typename number, typename VectorizedArrayType>
  VectorizedArrayType
  mask_function(const MaskFunctionType                         mask_function_type,
                const dealii::Point<dim, VectorizedArrayType> &location,
                const DEMParticleAccessor<dim, number>        &particle)
  {
    switch (mask_function_type)
      {
        case MaskFunctionType::discontinuous:
          return discontinuous_mask_function<dim, number, VectorizedArrayType>(location, particle);
        default:
          AssertThrow(false, dealii::ExcMessage("Unknown mask function type."));
      }
    return VectorizedArrayType(0.);
  }
} // namespace MeltPoolDG
