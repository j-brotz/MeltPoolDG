/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class LaserOperation
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const ScratchData<dim> &scratch_data;

    // Laser parameters
    const LaserData<double> laser_data;

    // Material parameters
    const MaterialData<double> material;

    // Laser position defined as spot center of the laser beam
    Point<dim> laser_position;

    // Current time
    double current_time;

    // Current intensity of the laser (between 0 and 1)
    double laser_intensity;

    // Requested laser model
    std::shared_ptr<LaserHeatSourceBase<dim>> laser_heat_source_operation;

  public:
    LaserOperation(const ScratchData<dim> &    scratch_data_in,
                   const LaserData<double> &   laser_data_in,
                   const MaterialData<double> &material_data_in);

    /**
     * Compute either the @p heat_source vector or @p heat_user_rhs, which is
     * the weak form of the heat_source, both with the DoFHandler layout provided
     * by @p temp_hanging_nodes_dof_idx. The input parameter comprise the
     * level set indicator, @p level_set_as_heaviside and the corresponding
     * index of the DoFHandler @p ls_dof_idx.
     *
     * The optional parameter @p zero_out specifies whether to clear out the to be filled
     * vector @p heat_user_rhs or @p heat_source. Optionally, a normal vector field cannot
     * be provided by @p normal_vector and @p normal_dof_idx. Alternatively,
     * the unit normal to the surface will be computed by the gradient of
     * @p level_set_as_heaviside.
     */
    void
    compute_heat_source(VectorType &           heat_source,
                        VectorType &           heat_user_rhs,
                        const VectorType &     level_set_as_heaviside,
                        const unsigned int     ls_dof_idx,
                        const unsigned int     temp_hanging_nodes_dof_idx,
                        const bool             zero_out       = false,
                        const BlockVectorType *normal_vector  = nullptr,
                        const unsigned int     normal_dof_idx = 0) const;

    /**
     * Reset the time.
     */
    void
    reset(const double start_time);

    /**
     * Move the laser position according to the provided scan speed for a time interface_value
     * @p dt.
     */
    void
    move_laser(double dt);

    /**
     * Getter function for the laser position.
     */
    const Point<dim> &
    get_laser_position() const;

    /**
     * Getter function for the current laser power.
     */
    double
    get_laser_power() const;

    /**
     * Getter function for the current laser impact type (volumetric, diffuse interface, sharp
     * interface).
     */
    LaserImpactType
    get_laser_impact_type() const;

  private:
    /**
     * Print info on current laser features.
     */
    void
    print() const;

    /**
     * Compute the time-dependent laser intensity (between 0 and 1) from the user-provided
     * parameters.
     */
    void
    compute_laser_intensity();
  };
} // namespace MeltPoolDG::Heat
