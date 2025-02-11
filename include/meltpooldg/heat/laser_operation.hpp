/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_heat_source_projection_based.hpp>
#include <meltpooldg/heat/laser_heat_source_volumetric.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_operation.hpp>

#include <functional>
#include <memory>
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

    const ScratchData<dim>                &scratch_data;
    const PeriodicBoundaryConditions<dim> &periodic_bc;

    // Laser parameters
    const LaserData<double> &laser_data;

    // current time
    double current_time;
    // current intensity of the laser (between 0 and 1)
    double laser_intensity;
    double current_power;
    // current laser position defined as spot center of the laser beam
    Point<dim> laser_position;

    std::shared_ptr<Function<dim, double>> intensity_profile;

    // Requested laser model
    std::unique_ptr<LaserHeatSourceVolumetric<dim>>      laser_heat_source_operation_volumetric;
    std::unique_ptr<LaserHeatSourceProjectionBased<dim>> laser_heat_source_operation_projection;

    // RTE
    std::unique_ptr<RadiativeTransport::RadiativeTransportOperation<dim>> rte_operation;
    std::unique_ptr<DoFHandler<dim>>                                      rte_dof_handler;
    std::unique_ptr<AffineConstraints<double>>                            rte_constraints_dirichlet;
    std::unique_ptr<AffineConstraints<double>> rte_hanging_node_constraints;
    std::map<dealii::types::boundary_id, std::shared_ptr<Function<dim>>>
                 rte_dirichlet_boundary_condition;
    unsigned int rte_dof_idx;
    unsigned int rte_hanging_nodes_dof_idx;
    unsigned int rte_quad_idx;

  public:
    LaserOperation(
      ScratchData<dim>                                         &scratch_data_in,
      const PeriodicBoundaryConditions<dim>                    &periodic_bc_in,
      const LaserData<double>                                  &laser_data_in,
      const VectorType                                         *heaviside_in         = nullptr,
      const unsigned int                                        hs_dof_idx_in        = 0,
      const RadiativeTransport::RadiativeTransportData<double> *rad_trans_data_in    = nullptr,
      const bool                                                problem_is_melt_pool = false,
      const bool                                                heat_is_cut          = false,
      const bool material_two_phase_transition_is_diffuse                            = false,
      const bool print_boundary_ids                                                  = false);

    void
    distribute_dofs(const FiniteElementData &fe_data);

    void
    setup_constraints();

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

    /**
     * Getter function for the underlying intensity profile.
     */
    std::shared_ptr<const Function<dim, double>>
    get_intensity_profile() const;

  private:
    /**
     * Print info on current laser features.
     */
    void
    print() const;

    /**
     * Compute the time-dependent laser intensity (between 0 and 1) from the user-provided
     * parameters.
     *
     * Return true if there is a change
     */
    bool
    compute_laser_intensity();
  };
} // namespace MeltPoolDG::Heat
