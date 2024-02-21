#include <deal.II/base/exceptions.h>

#include <meltpooldg/heat/laser_heat_source_uniform.hpp>
#include <meltpooldg/heat/laser_utilities.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserHeatSourceUniform<dim>::LaserHeatSourceUniform(
    const Tensor<1, dim, double>                      &laser_direction_in,
    const DeltaApproximationPhaseWeightedData<double> &delta_approximation_phase_weighted_data)
    : LaserHeatSourceBase<dim>(delta_approximation_phase_weighted_data)
    , laser_direction(laser_direction_in)
  {}

  template <int dim>
  double
  LaserHeatSourceUniform<dim>::local_compute_volumetric_heat_source(
    const Point<dim> & /*position*/,
    const Point<dim> & /*laser_position*/,
    const double /*power_density*/) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  template <int dim>
  double
  LaserHeatSourceUniform<dim>::local_compute_interfacial_heat_source(
    const Point<dim> & /*position*/,
    const Point<dim> & /*laser_position*/,
    const double                  power_density,
    const Tensor<1, dim, double> &normal_vector,
    const double                  delta_value,
    const double /*heaviside*/) const
  {
    return compute_projection_factor(laser_direction, normal_vector) * delta_value * power_density;
  }

  template class LaserHeatSourceUniform<1>;
  template class LaserHeatSourceUniform<2>;
  template class LaserHeatSourceUniform<3>;
} // namespace MeltPoolDG::Heat
