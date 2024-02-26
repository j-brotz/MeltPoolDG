#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <meltpooldg/heat/laser_data.hpp>

#include <cmath>

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

  template <int dim, typename number>
  class LaserIntensityProfileBase
  {
  public:
    virtual number
    compute_intensity(const Point<dim, number> &p) const = 0;

    virtual void
    update_power(const number power_in) = 0;
  };

  template <int dim, typename number>
  class UniformIntensityProfile : public LaserIntensityProfileBase<dim, number>
  {
  public:
    UniformIntensityProfile(const number power_in)
      : power(power_in)
    {}

    inline number
    compute_intensity(const Point<dim, number> &) const final
    {
      return power;
    }

    void
    update_power(const number power_in) final
    {
      power = power_in;
    }

  private:
    number power;
  };

  /**
   * Gauss laser profile for projection-based laser models
   *
   * intensity = power / ( radius^2 * pi/2 ) * exp( -2 * ( distance / radius )^2 )
   *
   * where the distance is between the point @param p and the laser beam center line defined by the
   * @param laser_position and the @param laser_direction.
   */
  template <int dim, typename number>
  class GaussProjectionIntensityProfile : public LaserIntensityProfileBase<dim, number>
  {
  public:
    GaussProjectionIntensityProfile(const number                  power,
                                    const number                  radius_in,
                                    const Point<dim, number>     &laser_position_in,
                                    const Tensor<1, dim, number> &laser_direction_in)
      : radius(radius_in)
      , laser_position(laser_position_in)
      , laser_direction(laser_direction_in)
    {
      AssertThrow(radius > 0.0,
                  ExcMessage("The laser beam radius must be greater than zero! Abort.."));

      update_power(power);
    }

    inline number
    compute_intensity(const Point<dim, number> &p) const final
    {
      const number s = compute_distance_to_line(p, laser_position, laser_direction) / radius;
      return peak_power_density * std::exp(-2. * s * s);
    }

    void
    update_power(const number power) final
    {
      peak_power_density = power / (radius * radius * numbers::PI_2);
    }


  private:
    const number                  radius;
    const Point<dim, number>     &laser_position;
    const Tensor<1, dim, number> &laser_direction;
    number                        peak_power_density;
  };

  /**
   * Volumetric spherical Gauss intensity profile
   *
   * intensity = power / ( radius * sqrt(pi/2) )^3 * exp( -2 * ( distance / radius )^2 )
   *
   * where the distance is between the point @param p and the laser beam center defined by the @param laser_position
   */
  template <int dim, typename number>
  class GaussVolumetricIntensityProfile : public LaserIntensityProfileBase<dim, number>
  {
  public:
    GaussVolumetricIntensityProfile(const number              power,
                                    const number              radius_in,
                                    const Point<dim, number> &laser_position_in)
      : radius(radius_in)
      , laser_position(laser_position_in)
    {
      AssertThrow(radius > 0.0,
                  ExcMessage("The laser beam radius must be greater than zero! Abort.."));

      update_power(power);
    }

    inline number
    compute_intensity(const Point<dim, number> &p) const final
    {
      const number s = p.distance(laser_position) / radius;
      return peak_power_density * std::exp(-2. * s * s);
    }

    void
    update_power(const number power) final
    {
      peak_power_density = power / Utilities::fixed_power<3>(radius * std::sqrt(numbers::PI_2));
    }

  private:
    const number              radius;
    const Point<dim, number> &laser_position;
    number                    peak_power_density;
  };

  /**
   * This class implements the laser heat source model of Gusarov et al. (2009).
   * DOI: 10.1115/1.3109245
   *
   *   ^ dim-1             2*R
   *   |                  <--->
   *   ----> x            | | |
   *                      | | |  laser beam
   *                      | | |     power
   *                      | | |
   *                      | | |
   *
   *    ---------------------------------------^
   *             ++                  ++        |
   *               ++      q       ++          |  layer thickness
   *                 +++        +++            |
   *                     ++++++                v
   *
   * The laser beam is direction is assumed to be in the negative dim-1 direction.
   */
  template <int dim, typename number>
  class GusarovIntensityProfile : public LaserIntensityProfileBase<dim, number>
  {
  public:
    GusarovIntensityProfile(const LaserData<number>::GusarovData &data_in,
                            const number                          power_in,
                            const number                          radius_in,
                            const Point<dim, number>             &laser_position_in)
      : data(data_in)
      , power(power_in)
      , radius(radius_in)
      , laser_position(laser_position_in)
      , lambda(data.extinction_coefficient * data.layer_thickness)
      , a(std::sqrt(1. - data.reflectivity))
      , D((1. - a) * (1. - a - data.reflectivity * (1. + a)) * std::exp(-2. * a * lambda) -
          (1. + a) * (1. + a - data.reflectivity * (1. - a)) * std::exp(2. * a * lambda))
    {
      AssertThrow(radius > 0.0,
                  ExcMessage("The laser beam radius must be greater than zero! Abort.."));
    }

    inline number
    compute_intensity(const Point<dim, number> &p) const final
    {
      const number distance = p.distance(laser_position);
      const number z        = -(p[dim - 1] - laser_position[dim - 1]);
      const number xi       = z * data.extinction_coefficient;

      return (z >= data.layer_thickness) || (z < laser_position[dim - 1]) ?
               0. :
               -data.extinction_coefficient * power_density(distance) * dq_dxi(xi);
    }

    void
    update_power(const number power_in) final
    {
      power = power_in;
    }

  private:
    inline number
    power_density(const number distance) const
    {
      return distance <= radius ? 3. * power / (numbers::PI * radius * radius) *
                                    Utilities::fixed_power<2>(1. - distance / radius) *
                                    Utilities::fixed_power<2>(1 + distance / radius) :
                                  0.0;
    }

    inline number
    dq_dxi(const number xi) const
    {
      const number &rho = data.reflectivity;

      // clang-format off
      return xi < lambda ?
             ((3 - 3 * rho) *(std::exp(-xi) + rho * std::exp(xi - 2 * lambda)))
             / // ------------------------------------------------------
             (4 * rho - 3)
             +
             2 * a * a * rho
             / // ------------------
             (D * (4 * rho - 3))
             *
             (
                     std::exp(-lambda) * (1 - rho * rho) *
                     (std::exp(-2 * a * xi) * (a - 1) + std::exp(2 * a * xi) * (a + 1))
                     -
                     (rho * std::exp(-2 * lambda) + 3) *
                     (std::exp(2 * a * (xi - lambda)) * (1 - a - rho * (a + 1)) -
                      std::exp(2 * a * (lambda - xi)) * (a + rho * (a - 1) + 1))
             ) :
             0.0;
      // clang-format on
    }

    const LaserData<number>::GusarovData &data;
    number                                power;
    const number                          radius;
    const Point<dim, number>             &laser_position;
    const number                          lambda;
    const number                          a;
    const number                          D;
  };

} // namespace MeltPoolDG::Heat