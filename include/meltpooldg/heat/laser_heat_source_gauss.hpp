
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/parameters.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * Spherical gauss laser heat source.
   *
   * local volumetric heat source p (W/m^3):
   * p = absorptivity * p_peak * exp( -2 ||x - laser_center||^2 / laser_radius^2 ) [1]
   *
   * with peak laser power density p_peak:
   * p_peak = laser_power / ( laser_radius * sqrt(pi/2) )^3
   *
   * [1] Meier, C., Fuchs, S. L., Hart, A. J., & Wall, W. A. (2021). A novel smoothed particle
   * hydrodynamics formulation for thermo-capillary phase change problems with focus on metal
   * additive manufacturing melt pool modeling. Computer Methods in Applied Mechanics and
   * Engineering, 381, 113812. https://doi.org/10.1016/j.cma.2021.113812
   */
  template <int dim>
  class LaserHeatSourceGauss : public LaserHeatSourceBase<dim>
  {
  public:
    LaserHeatSourceGauss(const LaserData<double>::GaussData &data_in);

    /**
     * volumetric heat source
     */
    double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         double            power) const final;

  private:
    /*
     * Laser power density.
     * returns: power * exp( -2 radius^2 / laser_radius^2 ) / ( laser_radius * sqrt(pi/2) )^3
     */
    double
    power_density(double radius, double power) const;

    const LaserData<double>::GaussData data;

    /*
     * Factor between peak power density and total laser power:
     * p_peak = laser_power * peak_power_density_factor
     *
     * peak_power_density_factor = 1 / ( laser_radius * sqrt(pi/2) )^3
     *
     * So that laser_power = int_R^3 p dx
     */
    const double peak_power_density_factor;
  };
} // namespace MeltPoolDG::Heat
