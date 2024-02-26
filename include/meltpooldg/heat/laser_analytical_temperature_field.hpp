/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February/June 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/material/material_data.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class LaserAnalyticalTemperatureField
  {
    using VectorType = LinearAlgebra::distributed::Vector<double>;

  public:
    static void
    compute_temperature_field(const ScratchData<dim>     &scratch_data,
                              const MaterialData<double> &material,
                              const LaserData<double>    &laser_data,
                              const double                laser_power,
                              const Point<dim>           &laser_position,
                              VectorType                 &temperature,
                              const VectorType           &level_set_as_heaviside,
                              const unsigned int          temp_dof_idx);

  private:
    /**
     * The temperature function below is derived from the publication
     * "Heat Source Modeling in Selective Laser Melting" by E. Mirkoohi, D. E. Seivers,
     *  H. Garmestani and S. Y. Liang.
     */
    static double
    local_compute_temperature_field(const MaterialData<double> &material,
                                    const LaserData<double>    &laser_data,
                                    const Point<dim>           &point,
                                    const double                heaviside,
                                    const double                scan_speed,
                                    const double                laser_power,
                                    const Point<dim>           &laser_position);
  };
} // namespace MeltPoolDG::Heat
