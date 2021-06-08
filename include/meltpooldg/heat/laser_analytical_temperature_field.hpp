/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February/June 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class LaserAnalyticalTemperatureField
  {
  public:
    LaserAnalyticalTemperatureField(const ScratchData<dim> &                 scratch_data,
                                    const LaserData<double>::AnalyticalData &data_in,
                                    const MaterialData<double> &             material_in,
                                    const double                             scan_speed,
                                    const unsigned int                       temp_dof_idx);

    void
    compute_temperature_field(const VectorType &level_set_as_heaviside,
                              VectorType &      temperature,
                              const double &    laser_power,
                              const Point<dim> &laser_position) const;

  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;
    /**
     * The temperature function below is derived from the publication on
     * "Heat Source Modeling in Selective Laser Melting" by E. Mirkoohi, D. E. Seivers,
     *  H. Garmestani and S. Y. Liang
     */
    double
    local_compute_temperature_field(const Point<dim> &point,
                                    const double      heaviside,
                                    const double      laser_power,
                                    const Point<dim> &laser_position) const;

    const ScratchData<dim> &                 scratch_data;
    const LaserData<double>::AnalyticalData &laser_data;
    const MaterialData<double> &             material;
    const double                             scan_speed;
    const unsigned int                       temp_dof_idx;
  };
} // namespace MeltPoolDG::Heat
