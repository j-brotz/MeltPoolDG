#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <meltpooldg/heat/laser_heat_source_gauss.hpp>
#include <meltpooldg/heat/laser_utilities.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>

#include <cmath>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  LaserHeatSourceGauss<dim>::LaserHeatSourceGauss(
    const LaserData<double>::GaussData                &data_in,
    const Tensor<1, dim, double>                      &laser_direction_in,
    const TwoPhaseFluidPropertiesTransitionType       &variable_properties_over_interface,
    const DeltaApproximationPhaseWeightedData<double> &delta_approximation_phase_weighted_data)
    : LaserHeatSourceBase<dim>(delta_approximation_phase_weighted_data)
    , data(data_in)
    , laser_direction(laser_direction_in)
    , variable_properties_over_interface(variable_properties_over_interface)
    , vol_peak_power_density_factor(
        1. / Utilities::fixed_power<3>(data.laser_beam_radius * std::sqrt(numbers::PI / 2)))
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
    // compute the projected distance from the laser center line, defined by the laser_position and
    // the laser_direction.
    const auto distance_normal_to_laser_axis =
      compute_distance_to_line(position, laser_position, laser_direction);

    const auto projection_factor = compute_projection_factor(laser_direction, normal_vector);

    const double weight =
      (variable_properties_over_interface != TwoPhaseFluidPropertiesTransitionType::sharp) ?
        heaviside :
        ((heaviside > 0.5) ? 1.0 : 0.0);

    const double absorptivity =
      LevelSet::Tools::interpolate(weight, data.absorptivity_gas, data.absorptivity_liquid);

    return absorptivity * projection_factor * delta_value *
           power_density_interfacial(distance_normal_to_laser_axis, power);
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
