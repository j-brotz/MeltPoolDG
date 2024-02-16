/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/boundary_conditions.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/rte_operation.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class LaserOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

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

    // RTE
    std::shared_ptr<RadiativeTransport::RadiativeTransportOperation<dim>> rte_operation;
    DoFHandler<dim>                                                       rte_dof_handler;
    AffineConstraints<double>                                             rte_constraints_dirichlet;
    AffineConstraints<double> rte_hanging_node_constraints;
    unsigned int              rte_dof_idx;
    unsigned int              rte_hanging_nodes_dof_idx;
    unsigned int              rte_quad_idx;

  public:
    LaserOperation(ScratchData<dim>         &scratch_data_in,
                   const Parameters<double> &data_in,
                   const VectorType         *heaviside_in  = nullptr,
                   const unsigned int        hs_dof_idx_in = 0);

    void
    distribute_dofs(const BaseData<double> &base_data);

    void
    setup_constraints(
      ScratchData<dim> &mutable_scratch_data,
      const std::function<const DirichletBoundaryConditions<dim> &(const std::string &)>
                                            &dirichlet_bc,
      const PeriodicBoundaryConditions<dim> &periodic_bc);

    void
    distribute_constraints();

    void
    reinit();

    void
    attach_vectors(std::vector<std::pair<const DoFHandler<dim> *,
                                         std::function<void(std::vector<VectorType *> &)>>> &data);

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    /**
     * Compute either the @p heat_source vector (for LaserImpactType::interface and
     * LaserImpactType::volumetric) or @p heat_user_rhs (for LaserImpactType::interface_sharp),
     * which is the weak form of the heat_source, both with the DoF layout provided
     * by @p temp_hanging_nodes_dof_idx. The input parameter comprise the
     * level set indicator, @p level_set_as_heaviside and the corresponding
     * index of the DoFHandler @p ls_dof_idx.
     *
     * The optional parameter @p zero_out specifies whether to clear out the to be filled
     * vector @p heat_source or @p heat_user_rhs. Optionally, a normal vector field can
     * be provided by @p normal_vector and @p normal_dof_idx. Alternatively,
     * the unit normal to the surface will be computed by the gradient of
     * @p level_set_as_heaviside.
     */
    void
    compute_heat_source(VectorType            &heat_source,
                        VectorType            &heat_user_rhs,
                        const VectorType      &level_set_as_heaviside,
                        const unsigned int     ls_dof_idx,
                        const unsigned int     temp_hanging_nodes_dof_idx,
                        const unsigned int     temp_quad_idx,
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
