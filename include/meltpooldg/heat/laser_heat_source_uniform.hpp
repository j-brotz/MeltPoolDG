
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * Uniform laser heat source.
   *
   * p = laser_power * delta_function
   *
   * The z-axis (= axis of the laser beam) is assumed to correspond to negative dim-1 coordinate.
   *
   * @Note: the parameter laser power is used as a power density here. In the case of interfacial
   * laser the power density is by area and in the case of volumetric laser it's by volume.
   */
  template <int dim>
  class LaserHeatSourceUniform : public LaserHeatSourceBase<dim>
  {
  public:
    LaserHeatSourceUniform(
      const DeltaApproximationPhaseWeightedData<double> &delta_approximation_phase_weighted_data);

    /**
     * Compute a DoF vector of the heat source for interface laser.
     */
    void
    compute_interfacial_heat_source(VectorType &            heat_source_vector,
                                    const ScratchData<dim> &scratch_data,
                                    const unsigned int      temp_dof_idx,
                                    const double            laser_power_density,
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
                                         const double      power_density) const final;

  private:
    double
    local_compute_interfacial_heat_source(const double                  power_density,
                                          const Tensor<1, dim, double> &normal_vector,
                                          const double                  delta_value,
                                          const double                  heaviside) const;

    std::unique_ptr<const DeltaApproximationBase<double>> delta_phase_weighted;
  };
} // namespace MeltPoolDG::Heat
