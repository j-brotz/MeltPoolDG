#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>
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
    LaserHeatSourceBase();

    LaserHeatSourceBase(const LevelSet::DeltaApproximationPhaseWeightedData<double>
                          &delta_approximation_phase_weighted_data);

    /**
     * Compute a DoF vector of the volumetric heat source.
     */
    void
    compute_volumetric_heat_source(VectorType             &heat_source_vector,
                                   const ScratchData<dim> &scratch_data,
                                   const unsigned int      temp_dof_idx,
                                   const double            laser_power,
                                   const Point<dim>       &laser_position,
                                   const bool              zero_out = true) const;

    /**
     * Compute a DoF vector of the interfacial heat source.
     */
    void
    compute_interfacial_heat_source(VectorType             &heat_source_vector,
                                    const ScratchData<dim> &scratch_data,
                                    const unsigned int      temp_dof_idx,
                                    const double            laser_power,
                                    const Point<dim>       &laser_position,
                                    const VectorType       &level_set_heaviside,
                                    const unsigned int      ls_dof_idx,
                                    const bool              zero_out       = true,
                                    const BlockVectorType  *normal_vector  = nullptr,
                                    const unsigned int      normal_dof_idx = 0) const;

    /**
     * Compute a DoF vector of the interfacial heat source by evaluating the surface
     * integral.
     */
    void
    compute_interfacial_heat_source_sharp(VectorType             &heat_rhs,
                                          const ScratchData<dim> &scratch_data,
                                          const unsigned int      temp_dof_idx,
                                          const double            laser_power,
                                          const Point<dim>       &laser_position,
                                          const VectorType       &level_set_heaviside,
                                          const unsigned int      ls_dof_idx,
                                          const bool              zero_out       = true,
                                          const BlockVectorType  *normal_vector  = nullptr,
                                          const unsigned int      normal_dof_idx = 0) const;

    /**
     * For a given finite element discretization, hidden within @p scratch_data,
     * compute a right-hand-side contribution for heat-transfer and store it
     * into the DoF-vector @p heat_rhs (with the corresponding index @p temp_dof_idx).
     * Based on the level-set indicator function (@p level_set_heaviside and
     * corresponding DoFHandler index @p ls_dof_idx), cells
     * will be categorized into the different phases. We loop over all element
     * faces that are along the phase boundary and perform a surface integral
     * based on the quadrature rule, given by the index @p temp_quad_idx,
     * to compute the laser impact. The laser parameters are @p laser_power and
     * @p laser_position. Set @p zero_out to true to zero out the right-hand-side
     * vector @p heat_rhs in advance. The optional arguments
     * @p normal_vector and @p normal_dof_idx, can be used to compute the unit
     * normal at the surface.
     *
     * @note This function should only be used, if the level-set isosurface
     * is aligned with element faces and does not cross the cells.
     *
     */
    void
    compute_interfacial_heat_source_sharp_conforming(VectorType             &heat_rhs,
                                                     const ScratchData<dim> &scratch_data,
                                                     const unsigned int      temp_dof_idx,
                                                     const unsigned int      temp_quad_idx,
                                                     const double            laser_power,
                                                     const Point<dim>       &laser_position,
                                                     const VectorType       &level_set_heaviside,
                                                     const unsigned int      ls_dof_idx,
                                                     const bool              zero_out,
                                                     const BlockVectorType  *normal_vector,
                                                     const unsigned int      normal_dof_idx) const;

  private:
    /**
     * Local volumetric heat source
     */
    virtual double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         double            power) const = 0;
    /**
     * Local interfacial heat source
     *
     * For the sharp interfacial heat source, set @p delta_value to 1.
     */
    virtual double
    local_compute_interfacial_heat_source(const Point<dim>             &position,
                                          const Point<dim>             &laser_position,
                                          const double                  power,
                                          const Tensor<1, dim, double> &normal_vector,
                                          const double                  delta_value,
                                          const double                  heaviside) const = 0;

    std::unique_ptr<const LevelSet::DeltaApproximationBase<double>> delta_phase_weighted;
  };
} // namespace MeltPoolDG::Heat
