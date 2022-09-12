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
                                   const unsigned int      temp_dof_idx,
                                   const double            laser_power,
                                   const Point<dim> &      laser_position,
                                   const bool              zero_out = true) const;
    /**
     * Local volumetric heat source
     */
    virtual double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         double            power) const = 0;

    /**
     * Compute a DoF vector of the interfacial heat source.
     */
    virtual void
    compute_interfacial_heat_source(VectorType &            heat_source_vector,
                                    const ScratchData<dim> &scratch_data,
                                    const unsigned int      temp_dof_idx,
                                    const double            laser_power,
                                    const Point<dim> &      laser_position,
                                    const VectorType &      level_set_heaviside,
                                    const unsigned int      ls_dof_idx,
                                    const bool              zero_out       = true,
                                    const BlockVectorType * normal_vector  = nullptr,
                                    const unsigned int      normal_dof_idx = 0) const = 0;

    /**
     * Compute a DoF vector of the interfacial heat source by evaluating the surface
     * integral.
     */
    virtual void
    compute_interfacial_heat_source_sharp(VectorType & /*heat_rhs*/,
                                          const ScratchData<dim> & /*scratch_data*/,
                                          const unsigned int /*temp_dof_idx*/,
                                          const double /*laser_power*/,
                                          const Point<dim> & /*laser_position*/,
                                          const VectorType & /*level_set_heaviside*/,
                                          const unsigned int /*ls_dof_idx*/,
                                          const bool /*zero_out*/                   = true,
                                          const BlockVectorType * /*normal_vector*/ = nullptr,
                                          const unsigned int /*normal_dof_idx*/     = 0) const
    {
      AssertThrow(false, ExcNotImplemented());
    }
  };
} // namespace MeltPoolDG::Heat
