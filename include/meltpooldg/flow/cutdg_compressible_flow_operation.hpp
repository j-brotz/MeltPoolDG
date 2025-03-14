/**
 * @brief This operator solves the compressible Navier-Stokes equations, comprising
 * the primary variables
 *  - density (ρ)
 *  - momentum (ρ u)
 *  - volume-specific energy (ρ E)
 *  using the cutDG method for single-phase problems.
 *
 * It is an extension of deal.II step-67 and is based on
 *
 * Fehn, N., Wall, W. A., & Kronbichler, M. (2019). A matrix‐free high‐order
 * discontinuous Galerkin compressible Navier‐Stokes solver: A performance
 * comparison of compressible and incompressible formulations for turbulent
 * incompressible flows. International Journal for Numerical Methods in Fluids,
 * 89(3), 71-102.
 *
 * and
 *
 * Ritthaler, A. (2024). A matrix-free cutDG formulation for complex flows,
 * Master's Thesis.
 */

#pragma once

#include <deal.II/base/function.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/solution_transfer.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/cutdg_compressible_flow_operator.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operation.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>
#include <variant>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number = double>
  class CutDGCompressibleFlowOperation
  {
    using VectorType            = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

   using CutFlowOperatorVariant = std::variant<
    CutDGCompressibleFlowOperator<dim, number, true>,
    CutDGCompressibleFlowOperator<dim, number, false>>;

  public:
    /**
     * Constructor.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param comp_flow_data_in Reference to the compressible flow data struct used.
     * @param time_iterator_in Reference to the used time stepping.
     * @param reinit_matrix_free_in Reinit_matrix_free function, which is registered.
     * @param comp_flow_dof_idx_in Index of the used dof handler for solution in @p scratch_data_in.
     * @param level_set_dof_idx_in Index of the used dof handler for level-set in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     * @param level_set_in level-set dof vector.
     */
    CutDGCompressibleFlowOperation(
      const ScratchData<dim>                                     &scratch_data_in,
      const CompressibleFlowData                                 &comp_flow_data_in,
      const TimeIterator<number>                                 &time_iterator_in,
      const std::function<void(const dealii::DoFHandler<dim> &)> &reinit_matrix_free_in,
      unsigned int                                                comp_flow_dof_idx_in  = 0,
      unsigned int                                                level_set_dof_idx_in  = 0,
      unsigned int                                                comp_flow_quad_idx_in = 0,
      const VectorType                                           &level_set_in          = 0);

    /**
     * Set up the required internal data structures. After a call to this function the solve()
     * function of the class can be utilized.
     */
    void
    reinit();

    /**
     * Set a body force, e.g. gravity, specified by the passed function.
     *
     * @param body_force_in Function specifying the body force.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_body_force(std::unique_ptr<Function<dim>> body_force_in);

    /**
     * Compute the maximum time step size arising from the convective and viscous time step limits
     * and optionally print it to the console.
     *
     * @param do_print If true, the time step limit is printed to the console.
     */
    number
    compute_time_step_size(bool do_print = false) const;

    /**
     * Distribute dofs needed for a finite element type given in @p CompressibleFlowData.
     * A FECollection is created to distinguish between liquid phase, gas phase and intersected
     * elements.
     *
     * @param dof_handler Reference to the used DoFHandler object.
     */
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const;

    /**
     * Solves the compressible Navier-Stokes equations for a single time step.
     *
     * @param current_time Current time at t^n.
     * @param time_step Current time step size.
     */
    void
    solve(number current_time, number time_step);

    /**
     * Set the initial condition of the solution dof vector.
     *
     * @param function Given function for initial condition.
     */
    void
    set_initial_condition(const Function<dim> &function);

    /**
     * Set the inflow field function in the case of an unfitted inflow boundary.
     *
     * @note The function simply passes the function to the corresponding cut operator function.
     */
    void
    set_inflow_field_unfitted_boundary(std::shared_ptr<Function<dim>> &inflow_function);

    /**
     * Set the velocity function in the case of an unfitted (rigid) moving object.
     *
     * @note The function simply passes the function to the corresponding cut operator function.
     */
    void
    set_unfitted_object_velocity(std::shared_ptr<Function<dim>> &velocity_function);

    /**
     * Set the (standard compressible flow, i.e. non-cut specific) boundary condition. This funciton
     * takes to parameters. One defines the tyoe of the boundary condition and the other one on the
     * one hand defines the boundary (by the boundary id) at whcih the boundary condition is applied
     * and optionally a corresponding prescribed function for the boundary condition.
     *
     * @param boundary_condition Type of the boundary condition.
     * @param boundary_condition_function A map consisting of the boundary id and the corresponding
     * boundary codnition function.
     */
    void
    set_boundary_condition(
      CompressibleBoundaryConditionType boundary_condition,
      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        boundary_condition_function);

    /**
     * Getter functions.
     */
    const VectorType &
    get_solution() const;

    VectorType &
    get_solution();

    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

    typename CutDGCompressibleFlowOperation<dim, number>::CutFlowOperatorVariant
    static create_cut_flow_operator_variant(
    bool is_viscous,
    CompressibleFlowScratchData<dim, number> &flow_scratch_data,
    const MappingInfoType                    &mapping_info_surface_in,
    const MappingInfoVectorType              &mapping_info_cells_in,
    const MappingInfoVectorType              &mapping_info_faces_in)
   {
    if (is_viscous)
     return CutDGCompressibleFlowOperator<dim, number, true>(
         flow_scratch_data,
         mapping_info_surface_in,
         mapping_info_cells_in,
         mapping_info_faces_in);
    else
     return CutDGCompressibleFlowOperator<dim, number, false>(
         flow_scratch_data,
         mapping_info_surface_in,
         mapping_info_cells_in,
         mapping_info_faces_in);
   }

  private:
    CompressibleFlowScratchData<dim, number> flow_scratch_data;

    const TimeIterator<double> &time_iterator;

    const unsigned int level_set_dof_idx;
    const VectorType  &level_set;
    VectorType         rhs;

    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier_old;

    CutUtil::SolutionTransferOperator<dim, double>                     cut_solution_transfer;
    std::function<void(const dealii::DoFHandler<dim> &)>               reinit_matrix_free;
    std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> reinit_vector;

    dealii::FESystem<dim> fe_point_temp;
    const unsigned int    n_dofs_per_cell;

    MappingInfoType       mapping_info_surface;
    MappingInfoVectorType mapping_info_cells;
    MappingInfoVectorType mapping_info_faces;

    CutFlowOperatorVariant cut_flow_operator;

    /**
     * Adapt the dof layout and solution vector to a new interface position, which is defined by the
     * zero-level-set-isosurface. The function contains following steps:
     * - classify cells according to current interface position
     * - adapt DoFHandler and solution vectors according to new interface position, extrapolate new
     * DoF values via ghost-penalty extrapolation
     * - compute quadrature rules for intersected cells, intersected faces and the phase surface
     * - reinit matrix-free object, rhs and solution vectors
     */
    void
    adapt_to_new_interface_position();

    /**
     * Classify cells according to the current state of the level-set field.
     */
    void
    classify_cells() const;

    /**
     * Compute the convective time step limit for the current mesh and flow field.
     */
    number
    compute_convective_time_step_limit() const;

    /**
     * Compute the minimum density currently occurring in the flow field.
     */
    number
    compute_minimum_density() const;

    /**
     * Compute the quadrature rules for intersected elements, intersected faces and phase surfaces
     */
    void
    compute_intersected_quadrature();
  };

  //! inlined functions
  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  CutDGCompressibleFlowOperation<dim, number>::get_solution() const
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  CutDGCompressibleFlowOperation<dim, number>::get_solution()
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number>
  const dealii::DoFHandler<dim> &
  CutDGCompressibleFlowOperation<dim, number>::get_dof_handler() const
  {
    return flow_scratch_data.scratch_data.get_dof_handler(flow_scratch_data.dof_idx);
  }

} // namespace MeltPoolDG::Flow
