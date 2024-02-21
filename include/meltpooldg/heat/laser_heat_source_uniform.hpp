#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>

#include <memory>

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
   * @note The parameter laser power is used as a power density here. In the case of interfacial
   * laser the power density is by area and in the case of volumetric laser it's by volume.
   */
  template <int dim>
  class LaserHeatSourceUniform : public LaserHeatSourceBase<dim>
  {
  public:
    LaserHeatSourceUniform(const Tensor<1, dim, double> &laser_direction_in,
                           const LevelSet::DeltaApproximationPhaseWeightedData<double>
                             &delta_approximation_phase_weighted_data);

  private:
    /**
     * volumetric heat source
     */
    double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         const double      power_density) const final;

    /**
     * interfacial heat source
     */
    double
    local_compute_interfacial_heat_source(const Point<dim>             &position,
                                          const Point<dim>             &laser_position,
                                          const double                  power,
                                          const Tensor<1, dim, double> &normal_vector,
                                          const double                  delta_value,
                                          const double                  heaviside) const final;

    const Tensor<1, dim, double> &laser_direction;

    std::unique_ptr<const LevelSet::DeltaApproximationBase<double>> delta_phase_weighted;
  };
} // namespace MeltPoolDG::Heat
