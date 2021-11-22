
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * Spherical gauss laser heat source.
   *
   * Volumetric case:
   * local volumetric heat source p (W/m^3):
   * p = absorptivity * p_peak * exp( -2 ||x - laser_center||^2 / laser_radius^2 )
   *
   * with peak laser power density p_peak:
   * p_peak = laser_power / ( laser_radius * sqrt(pi/2) )^3
   *
   * Interface case:
   * p = absorptivity * < -interface_normal * laser_direction > * p_peak *
   * exp( -2 ||x - laser_center||^2 / laser_radius^2 ) delta_function [1]
   *
   * with peak laser power density p_peak:
   * p_peak = laser_power / ( laser_radius^2 * pi/2 )
   *
   * The z-axis (= axis of the laser beam) is assumed to correspond to negative dim-1 coordinate.
   *
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
    LaserHeatSourceGauss(
      const LaserData<double>::GaussData &         data_in,
      const TwoPhaseFluidPropertiesTransitionType &variable_properties_over_interface);

    /**
     * Compute a DoF vector of the heat source for interface laser.
     */
    void
    compute_interfacial_heat_source(VectorType &            heat_source_vector,
                                    const ScratchData<dim> &scratch_data,
                                    const unsigned int      temp_dof_idx,
                                    const double            laser_power,
                                    const Point<dim> &      laser_position,
                                    const VectorType &      level_set_heaviside,
                                    const unsigned int      ls_dof_idx,
                                    const bool              zero_out       = true,
                                    const BlockVectorType * normal_vector  = nullptr,
                                    const unsigned int      normal_dof_idx = 0) const final;

    /**
     * volumetric heat source
     */
    double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         const double      power) const final;

  private:
    /**
     * interface heat source
     */
    double
    local_compute_interfacial_heat_source(const Point<dim> &            position,
                                          const Point<dim> &            laser_position,
                                          const double                  power,
                                          const Tensor<1, dim, double> &normal_vector,
                                          const double                  delta_value,
                                          const double                  heaviside) const;
    /*
     * Laser power density in volumetric case.
     * returns: power * exp( -2 radius^2 / laser_radius^2 ) / ( laser_radius * sqrt(pi/2) )^3
     */
    double
    power_density_volumetric(const double radius, const double power) const;

    /*
     * Laser power density in interface case.
     * returns: power * exp( -2 radius^2 / laser_radius^2 ) / ( laser_radius^2 * pi/2 )
     */
    double
    power_density_interfacial(const double radius, const double power) const;

    const LaserData<double>::GaussData data;

    const TwoPhaseFluidPropertiesTransitionType variable_properties_over_interface;

    /*
     * Factor between peak power density and total laser power in volumetric case:
     * p_peak = laser_power * peak_power_density_factor
     *
     * vol_peak_power_density_factor = 1 / ( laser_radius * sqrt(pi/2) )^3
     *
     * So that laser_power = int_R^3 p dx
     */
    const double vol_peak_power_density_factor;

    /*
     * Factor between peak power density and total laser power in interface case:
     * p_peak = laser_power * peak_power_density_factor
     *
     * surf_peak_power_density_factor = 1 / ( laser_radius^2 * pi/2 )
     *
     * So that laser_power = int_R^2 p dx
     */
    const double surf_peak_power_density_factor;

    std::unique_ptr<const DeltaApproximationBase<double>> delta_phase_weighted;
  };
} // namespace MeltPoolDG::Heat
