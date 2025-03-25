#pragma once

#include <deal.II/base/point.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/utilities/material_data.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  class LaserAnalyticalTemperatureField
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    static void
    compute_temperature_field(const ScratchData<dim, dim, number> &scratch_data,
                              const MaterialData<number>          &material,
                              const LaserData<number>             &laser_data,
                              const number                         laser_power,
                              const dealii::Point<dim>            &laser_position,
                              VectorType                          &temperature,
                              const VectorType                    &level_set_as_heaviside,
                              const unsigned int                   heat_dof_idx);

  private:
    /**
     * The temperature function below is derived from the publication
     * "Heat Source Modeling in Selective Laser Melting" by E. Mirkoohi, D. E. Seivers,
     *  H. Garmestani and S. Y. Liang.
     */
    static number
    local_compute_temperature_field(const MaterialData<number> &material,
                                    const LaserData<number>    &laser_data,
                                    const dealii::Point<dim>   &point,
                                    const number                heaviside,
                                    const number                scan_speed,
                                    const number                laser_power,
                                    const dealii::Point<dim>   &laser_position);
  };
} // namespace MeltPoolDG::Heat
