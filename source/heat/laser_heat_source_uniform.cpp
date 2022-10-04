#include <meltpooldg/heat/laser_heat_source_uniform.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserHeatSourceUniform<dim>::LaserHeatSourceUniform(
    const DeltaApproximationPhaseWeightedData<double> &delta_approximation_phase_weighted_data)
    : LaserHeatSourceBase<dim>(delta_approximation_phase_weighted_data)
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
    // assume laser direction coincides with the negative dim-1 direction
    const double projection_factor = std::invoke([&]() {
      const double fac = normal_vector * -Point<dim>::unit_vector(dim - 1);
      if (fac < 0.0)
        return 0.0;
      return fac;
    });

    return projection_factor * delta_value * power_density;
  }

  template class LaserHeatSourceUniform<1>;
  template class LaserHeatSourceUniform<2>;
  template class LaserHeatSourceUniform<3>;
} // namespace MeltPoolDG::Heat
