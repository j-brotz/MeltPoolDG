#include <meltpooldg/heat/laser_heat_source_gauss.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserHeatSourceGauss<dim>::LaserHeatSourceGauss(
    const LaserData<double>::GaussData                &data_in,
    const TwoPhaseFluidPropertiesTransitionType       &variable_properties_over_interface,
    const DeltaApproximationPhaseWeightedData<double> &delta_approximation_phase_weighted_data)
    : LaserHeatSourceBase<dim>(delta_approximation_phase_weighted_data)
    , data(data_in)
    , variable_properties_over_interface(variable_properties_over_interface)
    , vol_peak_power_density_factor(
        1. / std::pow(data.laser_beam_radius * std::sqrt(numbers::PI / 2), 3))
    , surf_peak_power_density_factor(
        1. / (data.laser_beam_radius * data.laser_beam_radius * numbers::PI / 2))
  {
    AssertThrow(data.laser_beam_radius > 0.0,
                ExcMessage("The laser beam radius must be greater than zero! Abort.."));
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::local_compute_volumetric_heat_source(const Point<dim> &position,
                                                                  const Point<dim> &laser_position,
                                                                  const double      power) const
  {
    const double distance = position.distance(laser_position);
    return data.absorptivity_liquid * power_density_volumetric(distance, power);
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::local_compute_interfacial_heat_source(
    const Point<dim>             &position,
    const Point<dim>             &laser_position,
    const double                  power,
    const Tensor<1, dim, double> &normal_vector,
    const double                  delta_value,
    const double                  heaviside) const
  {
    Point<dim> projected_position(position);

    if (dim > 1)
      {
        // Calculate the projected distance in x (2D) or x and y (3D) direction.
        // To this end, we calculate a projected point lying in the laser plane.
        projected_position[dim - 1] = laser_position[dim - 1];
      }
    const double distance = laser_position.distance(projected_position);

    // assume laser direction coincides with the negative dim-1 direction
    double projection_factor = normal_vector * -Point<dim>::unit_vector(dim - 1);
    if (projection_factor < 0.0)
      projection_factor = 0.0;

    const double weight =
      (variable_properties_over_interface != TwoPhaseFluidPropertiesTransitionType::sharp) ?
        heaviside :
        ((heaviside > 0.5) ? 1.0 : 0.0);

    const double absorptivity =
      LevelSet::Tools::interpolate(weight, data.absorptivity_gas, data.absorptivity_liquid);

    return absorptivity * projection_factor * delta_value *
           power_density_interfacial(distance, power);
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::power_density_volumetric(const double radius, const double power) const
  {
    const double s          = radius / data.laser_beam_radius;
    const double peak_power = power * vol_peak_power_density_factor;
    return peak_power * std::exp(-2. * s * s);
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::power_density_interfacial(const double radius,
                                                       const double power) const
  {
    const double s          = radius / data.laser_beam_radius;
    const double peak_power = power * surf_peak_power_density_factor;
    return peak_power * std::exp(-2. * s * s);
  }

  template class LaserHeatSourceGauss<1>;
  template class LaserHeatSourceGauss<2>;
  template class LaserHeatSourceGauss<3>;
} // namespace MeltPoolDG::Heat
