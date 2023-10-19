#pragma once

#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/material/material_data.hpp>

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
      const LaserData<double>::GaussData                &data_in,
      const TwoPhaseFluidPropertiesTransitionType       &variable_properties_over_interface,
      const DeltaApproximationPhaseWeightedData<double> &delta_approximation_phase_weighted_data);

    /**
     * Compute a DoF vector and assemble it into @p heat_rhs, considering a Gaussian laser heat
     * source (see class description) based on a surface integral evaluation
     *
     *   /     \    (Ω)  /   (Γ) \    (Γ)
     *  | w, s | = N   s |  x     | JxW
     *   \     /    a    \   q   /     q
     *          Γ
     * with shape functions N_a^(Ω) (from the background mesh), the laser heat source function s(x)
     * quadrature points (from the interface mesh) x_q^(Γ) and integration weights JxW_q^(Γ)
     * (from the interface mesh). The geometric quantities for cell cut by the interface (Φ=0)
     * are visualized as follows
     *
     * +-----o-------------+
     * |    /              |
     * |   *               |
     * |  / (Γ)            |
     * | * x          Ω    |
     * |/   q              |
     * o                   |
     * +-------------------+
     *
     * whereas Ω denotes the background mesh and Γ the interface. The parameters comprise
     * @p scratch_data (DoF handlers, constraints, finite element info), the DoF index
     * of the temperature field @p temp_dof_idx, parameters for the heat source evaluation,
     * @p laser_power and @p laser_position, level-set related quantities @p level_set_heaviside
     * and @p ls_dof_idx. If @p zero_out is set, @p heat_rhs will be zeroed out befor the
     * operation. If a DoF vector for the @p normal_vector and a corresponding index
     * @p normal_dof_idx is given, the latter will be used for computing the laser heat
     * source.
     */

  private:
    /**
     * volumetric heat source
     */
    double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         const double      power) const final;
    /**
     * interface heat source
     */
    double
    local_compute_interfacial_heat_source(const Point<dim>             &position,
                                          const Point<dim>             &laser_position,
                                          const double                  power,
                                          const Tensor<1, dim, double> &normal_vector,
                                          const double                  delta_value,
                                          const double                  heaviside) const final;
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
  };
} // namespace MeltPoolDG::Heat
