#include <meltpooldg/heat/laser_heat_source_gusarov.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserHeatSourceGusarov<dim>::LaserHeatSourceGusarov(
    const LaserData<double>::GusarovData &gusarov_data_in)
    : gusarov_data(gusarov_data_in)
    , lambda(gusarov_data.extinction_coefficient * gusarov_data.layer_thickness)
    , a(std::sqrt(1. - gusarov_data.reflectivity))
    , D((1. - a) * (1. - a - gusarov_data.reflectivity * (1. + a)) * std::exp(-2. * a * lambda) -
        (1. + a) * (1. + a - gusarov_data.reflectivity * (1. - a)) * std::exp(2. * a * lambda))
  {}

  template <int dim>
  double
  LaserHeatSourceGusarov<dim>::local_compute_volumetric_heat_source(
    const Point<dim> &position,
    const Point<dim> &laser_position,
    const double      power) const
  {
    const double radius = position.distance(laser_position);
    const double z      = -(position[dim - 1] - laser_position[dim - 1]);
    const double xi     = z * gusarov_data.extinction_coefficient;

    return (z >= gusarov_data.layer_thickness) || (z < laser_position[dim - 1]) ?
             0. :
             -gusarov_data.extinction_coefficient * power_density(radius, power) * dq_dxi(xi);
  }

  template <int dim>
  double
  LaserHeatSourceGusarov<dim>::power_density(const double radius, const double power) const
  {
    const double &R = gusarov_data.laser_beam_radius;
    return radius <= R ?
             3. * power / (numbers::PI * R * R) * Utilities::fixed_power<2>(1. - radius / R) *
               Utilities::fixed_power<2>(1 + radius / R) :
             0.0;
  }

  template <int dim>
  double
  LaserHeatSourceGusarov<dim>::dq_dxi(const double xi) const
  {
    const double &rho = gusarov_data.reflectivity;

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

  template <int dim>
  double
  LaserHeatSourceGusarov<dim>::local_compute_interfacial_heat_source(
    const Point<dim> & /*position*/,
    const Point<dim> & /*laser_position*/,
    const double /*power*/,
    const Tensor<1, dim, double> & /*normal_vector*/,
    const double /*delta_value*/,
    const double /*heaviside*/) const
  {
    AssertThrow(
      false,
      ExcMessage("The Gurasov laser heat source model is not suited for surface impact! Abort..."));
    return 0.0;
  }

  template class LaserHeatSourceGusarov<1>;
  template class LaserHeatSourceGusarov<2>;
  template class LaserHeatSourceGusarov<3>;
} // namespace MeltPoolDG::Heat
