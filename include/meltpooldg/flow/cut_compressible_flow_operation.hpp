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
#include <meltpooldg/flow/compressible_flow_operation.hpp>
#include <meltpooldg/flow/cut_compressible_flow_operator.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim, typename number = double>
  class CutCompressibleFlowOperation final : public CompressibleFlowOperation<dim, number>
  {
    using VectorType            = typename CompressibleFlowOperation<dim, number>::VectorType;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

  public:
    /**
     * Constructor.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param comp_flow_data_in Reference to the compressible flow data struct used.
     * @param time_iterator_in Reference to the used time stepping.
     * @param comp_flow_dof_idx_in Index of the used dof handler for solution in @p scratch_data_in.
     * @param level_set_dof_idx_in Index of the used dof handler for level-set in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     * @param level_set_in level-set dof vector.
     */
    CutCompressibleFlowOperation(const ScratchData<dim>     &scratch_data_in,
                                 const CompressibleFlowData &comp_flow_data_in,
                                 const TimeIterator<double> &time_iterator_in,
                                 unsigned int                comp_flow_dof_idx_in  = 0,
                                 unsigned int                level_set_dof_idx_in  = 0,
                                 unsigned int                comp_flow_quad_idx_in = 0,
                                 const VectorType           &level_set_in          = 0);

    /**
     * Set up the required internal data structures. After a call to this function the solve()
     * function of the class can be utilized.
     */
    void
    reinit() override;

    /**
     * Distribute dofs needed for a finite element type given in @p CompressibleFlowData.
     * A FECollection is created to distinguish between liquid phase, gas phase and intersected
     * elements.
     *
     * @param dof_handler Reference to the used DoFHandler object.
     */
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const override;

    /**
     * Solves the compressible Navier-Stokes equations for a single time step.
     *
     * @param current_time Current time at t^n.
     * @param time_step Current time step size.
     */
    void
    solve(double current_time, double time_step) override;

    /**
     * Set the initial condition of the solution dof vector.
     *
     * @param function Given function for initial condition.
     */
    void
    set_initial_condition(const Function<dim> &function) override;

    /**
     * Set the inflow field function in the case of an unfitted inflow boundary.
     *
     * @note The function simply passes the function to the corresponding cut operator function.
     */
    void
    set_inflow_field_unfitted_boundary(std::shared_ptr<Function<dim>> &inflow_function) override;

    /**
     * Set the velocity function in the case of an unfitted (rigid) moving object.
     *
     * @note The function simply passes the function to the corresponding cut operator function.
     */
    void
    set_unfitted_object_velocity(std::shared_ptr<Function<dim>> &velocity_function) override;

    /**
     * Classify cells according to the current state of the level-set field.
     */
    void
    classify_cells() const;

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
     * Register the reinit_matrix_free lambda function.
     *
     * @param reinit_matrix_free_in  Reinit_matrix_free function, which is registered.
     */
    void
    register_reinit_matrix_free(
      std::function<void(const dealii::DoFHandler<dim> &)> reinit_matrix_free_in) override;

  private:
    const TimeIterator<double> &time_iterator;

    const unsigned int level_set_dof_idx;
    const VectorType  &level_set;
    VectorType         rhs;

    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier_old;

    CutUtil::SolutionTransferOperator<dim, double>                     cut_solution_transfer;
    std::function<void(const dealii::DoFHandler<dim> &)>               reinit_matrix_free;
    std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> reinit_vector;

    FESystem<dim>      fe_point_temp;
    const unsigned int n_dofs_per_cell;

    MappingInfoType       mapping_info_surface;
    MappingInfoVectorType mapping_info_cells;
    MappingInfoVectorType mapping_info_faces;

    LinearSolver linear_solver;

    /**
     * Compute the convective time step limit for the current mesh and flow field.
     */
    number
    compute_convective_time_step_limit() const override;

    /**
     * Compute the minimum density currently occurring in the flow field.
     */
    number
    compute_minimum_density() const override;

    /**
     * Compute the quadrature rules for intersected elements, intersected faces and phase surfaces
     */
    void
    compute_intersected_quadrature();
  };
} // namespace MeltPoolDG::Flow
