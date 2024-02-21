#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * Computes the projection factor for angled laser interface interactions.
   *
   * @param normal_vector the unit normal vector pointing inside the liquid domain
   */
  template <int dim, typename number>
  inline number
  compute_projection_factor(const Tensor<1, dim, number> &laser_direction,
                            const Tensor<1, dim, number> &normal_vector)
  {
    Assert(std::abs(laser_direction.norm() - 1.0) < 1e-8,
           ExcMessage("The laser direction must be a unit vector"));
    Assert(std::abs(normal_vector.norm() - 1.0) < 1e-8,
           ExcMessage("The laser direction must be a unit vector"));

    const number fac = normal_vector * laser_direction;
    if (fac < 0.0)
      return 0.0;
    return fac;
  }

  /**
   * Computes the minimum distance between the point @param p and the infinite line that is defined
   * by the @param line_base and the @param line_direction, which must be a unit vector.
   */
  template <int dim, typename number>
  inline number
  compute_distance_to_line(const Point<dim, number>     &p,
                           const Point<dim, number>     &line_base,
                           const Tensor<1, dim, number> &line_direction)
  {
    Assert(std::abs(line_direction.norm() - 1.0) < 1e-8,
           ExcMessage("The line direction must be a unit vector"));

    if (dim == 1)
      return 0.0;
    else if (dim == 2)
      return std::abs(cross_product_2d(p - line_base) * line_direction);
    else if (dim == 3)
      return cross_product_3d(p - line_base, line_direction).norm();
    else
      Assert(false, ExcNotImplemented());
    return 0.0;
  }
} // namespace MeltPoolDG::Heat