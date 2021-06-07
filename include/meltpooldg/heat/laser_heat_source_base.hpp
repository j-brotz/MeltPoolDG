#pragma once

#include <deal.II/base/point.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * Laser heat source model interface.
   */
  template <int dim>
  class LaserHeatSourceBase
  {
  public:
    /**
     * Compute a DoF vector of the volumetric heat source.
     */
    void
    compute_volumetric_heat_source(VectorType &            heat_source_vector,
                                   const ScratchData<dim> &scratch_data,
                                   unsigned int            temp_dof_idx,
                                   double                  laser_power,
                                   const Point<dim> &      laser_position,
                                   bool                    zero_out = true) const;

    /**
     * Compute a DoF vector of the interfacial heat source.
     */
    virtual void
    compute_interfacial_heat_source(VectorType &            heat_source_vector,
                                    const ScratchData<dim> &scratch_data,
                                    unsigned int            temp_dof_idx,
                                    double                  laser_power,
                                    const Point<dim> &      laser_position,
                                    const VectorType &      level_set_heaviside,
                                    unsigned int            ls_dof_idx,
                                    bool                    zero_out = true) const = 0;

    /**
     * Local volumetric heat source
     */
    virtual double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         double            power) const = 0;

    /**
     * Local interface heat source at two-phase interface
     */
    virtual double
    local_compute_interfacial_heat_source(const Point<dim> &            position,
                                          const Point<dim> &            laser_position,
                                          double                        power,
                                          const Tensor<1, dim, double> &normal_vector,
                                          double                        delta_value) const = 0;
  };
} // namespace MeltPoolDG::Heat
