#pragma once

#include <deal.II/base/point.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * Laser heat source model interface.
   */
  template <int dim>
  class LaserHeatSource
  {
  public:
    /**
     * Compute a DoF vector of the heat source.
     */
    virtual void
    compute_volumetric_heat_source(VectorType &            heat_source_vector,
                                   const ScratchData<dim> &scratch_data,
                                   unsigned int            temp_dof_idx,
                                   double                  laser_power,
                                   const Point<dim> &      laser_position,
                                   bool                    zero_out = true) const;

    /**
     * Local volumetric heat source
     */
    virtual double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         double            power) const = 0;
  };
} // namespace MeltPoolDG::Heat
