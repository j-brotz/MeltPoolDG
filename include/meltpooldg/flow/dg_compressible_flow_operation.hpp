/**
 * @brief This operation solves the compressible Navier-Stokes equations, comprising
 * the primary variables
 *  - density (ρ)
 *  - momentum (ρ u)
 *  - volume-specific energy (ρ E)
 *
 * It is an extension of deal.II step-67 and is based on the paper
 *
 * Fehn, N., Wall, W. A., & Kronbichler, M. (2019). A matrix‐free high‐order
 * discontinuous Galerkin compressible Navier‐Stokes solver: A performance
 * comparison of compressible and incompressible formulations for turbulent
 * incompressible flows. International Journal for Numerical Methods in Fluids,
 * 89(3), 71-102.
 */

#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include "meltpooldg/utilities/attach_vectors.hpp"
#include "meltpooldg/utilities/matrix_free_util.hpp"
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_scratch_data.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_base.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>

#include <memory>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Operation that performs a full time step for the compressible Navier-Stokes.
   */
  template <int dim, typename number>
  class DGCompressibleFlowOperation
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * Initializes all internal data structures required to simulate compressible Navier-Stokes
     * flows.
     *
     * @param scratch_data Reference to the used ScratchData object.
     * @param flow_data Reference to the compressible flow data struct used.
     * @param material_data_in Reference to the material data struct.
     */
    explicit DGCompressibleFlowOperation(
      const dealii::Triangulation<dim>                 &tria,
      ScratchData<dim, dim, number>                    &scratch_data,
      const CompressibleFlowData<number>               &flow_data,
      const CompressibleFluidMaterialPhaseData<number> &material_data_in);


    /**
     * @brief Constructor.
     *
     * Initializes all internal data structures required to simulate compressible Navier-Stokes
     * flows.
     *
     * @param scratch_data Reference to the used ScratchData object.
     * @param flow_data Reference to the compressible flow data struct used.
     * @param material_data_in Reference to the material data struct.
     * @param flow_dof_idx Index of the used dof handler in @p scratch_data.
     * @param flow_quad_idx Index of the used quadrature object in @p scratch_data.
     */
    explicit DGCompressibleFlowOperation(
      const ScratchData<dim, dim, number>              &scratch_data,
      const CompressibleFlowData<number>               &flow_data,
      const CompressibleFluidMaterialPhaseData<number> &material_data_in,
      unsigned int                                      flow_dof_idx  = 0,
      unsigned int                                      flow_quad_idx = 0);

    /**
     * @brief Set up the required internal data structures.
     *
     * After a call to this function the solve() function of the class can be utilized.
     */
    void
    reinit();

    /**
     * @brief Solves the compressible Navier-Stokes equations for a single time step.
     *
     * @param current_time Current time at t^n.
     * @param time_step Current time step size.
     */
    void
    solve(const number current_time, const number time_step);

    /**
     * @brief Distribute the degrees of freedom to the passed dof handler object.
     *
     * @param dof_handler Dof handler object used for the compressible flow solver.
     */
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const;

    /**
     * @brief Distribute the degrees of freedom to the object local dof handler and attach it to scratch data.
     */
    void
    distribute_dofs();

    /**
     * @brief Create the quadrature object and attach it to scratch data.
     */
    void
    create_quadrature();

    /**
     * @brief Create the required constraints and attach it to scratch data.
     */
    void
    create_constraints();

    void
    initialize_data_structures()
    {
      distribute_dofs();
      create_quadrature();
      create_constraints();
    }



    /**
     * @brief Set the boundary conditions.
     *
     * @param simulation_case dealii::Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     *
     * @note The function simply passes the parameters to the set_boundary_conditions function in the
     * CompressibleFlowBoundaryConditions class.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name);

    /**
     * @brief Set a body force, e.g. gravity, specified by the passed function.
     *
     * @param body_force_in Function specifying the body force.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_body_force(std::unique_ptr<dealii::Function<dim>> body_force_in);

    void
    add_external_force(
      std::shared_ptr<AdditionalCellAndQuadOperation<dim, number>>         external_force_residuum,
      std::shared_ptr<AdditionalCellAndQuadOperationJacobian<dim, number>> external_force_jacobian);

    /**
     * @brief Compute the maximum time step size.
     *
     * The maximum time step size arises from the convective and viscous time step limits.
     * Optionally, it is printed to the console.
     *
     * @param do_print If true, the time step limit is printed to the console.
     *
     * @return The computed maximum time step size.
     */
    number
    compute_time_step_size(bool do_print = false) const;

    /**
     * @brief Set the solution vector to the passed initial flow field state.
     *
     * @param function Initial condition of the flow field.
     */
    void
    set_initial_condition(const dealii::Function<dim> &function);

    /**
     * @brief Attach the solution to the passed data out object.
     *
     * The solution which are added are the density, the momentum and the energy density.
     *
     * @param data_out Object to which the solution vector is attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /**
     * @brief Constant getter function for the current solution vector.
     */
    const VectorType &
    get_solution() const;

    /**
     * @brief Getter function for the current solution vector.
     */
    VectorType &
    get_solution();

    /**
     * @brief Getter function for the current solution vector in primitive variables (pressure,
     * velocity, temperature).
     */
    VectorType &
    get_solution_in_primitive_variables();

    void
    attach_for_coarsening_and_refinement(
      DoFHandlerAndVectorDataType<dim, dealii::LinearAlgebra::distributed::Vector<number>> &in)
    {
      in.emplace_back(&flow_scratch_data.scratch_data.get_dof_handler(flow_scratch_data.dof_idx),
                      [&](
                        std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vec_in) {
                        for (auto &sol : flow_scratch_data.solution_history.get_all_solutions())
                          vec_in.push_back(&sol);
                      });
    }

    /**
     * @brief Constant getter function for the DoFHandler.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

    MatrixFreeContext<dim, number>
    get_matrix_free_context()
    {
      return {flow_scratch_data.scratch_data.get_matrix_free(),
              flow_scratch_data.dof_idx,
              flow_scratch_data.quad_idx};
    }

    CompressibleFluidMaterialPhaseData<number>
    get_phase_material_data()
    {
      return flow_scratch_data.material.data;
    }

  private:
    /// Scratch data for compressible flows
    CompressibleFlowScratchData<dim, number> flow_scratch_data;

    /// Compressible flow operator object
    std::unique_ptr<DGCompressibleFlowOperatorBase<dim, number>> comp_flow_operator;

    /// Solution vector in primitive variable formulation (pressure, velocity, temperature)
    VectorType solution_primitive_variables;

    ///
    dealii::DoFHandler<dim> dof_handler;

    dealii::AffineConstraints<number> constraints;

    /**
     * @brief Compute the convective time step limit for the current mesh and flow field.
     *
     * @return Maximum convective time step size.
     */
    number
    compute_convective_time_step_limit() const;

    /**
     * @brief Compute the minimum density currently occurring in the flow field.
     *
     * @return Minimum density.
     */
    number
    compute_minimum_density() const;

    /**
     * @brief Set up the operator to suit the specified time integration scheme.
     *
     * @param external_forces Pointer to a struct implementing external forces acting on the fluid.
     */
    void
    setup_operator();
  };


  //! inlined functions
  template <int dim, typename number>
  const dealii::LinearAlgebra::distributed::Vector<number> &
  DGCompressibleFlowOperation<dim, number>::get_solution() const
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number>
  dealii::LinearAlgebra::distributed::Vector<number> &
  DGCompressibleFlowOperation<dim, number>::get_solution()
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number>
  dealii::LinearAlgebra::distributed::Vector<number> &
  DGCompressibleFlowOperation<dim, number>::get_solution_in_primitive_variables()
  {
    update_primitive_variables_solution<dim, number>(solution_primitive_variables,
                                                     get_solution(),
                                                     flow_scratch_data.scratch_data,
                                                     flow_scratch_data.dof_idx,
                                                     flow_scratch_data.quad_idx,
                                                     &flow_scratch_data.material);
    return solution_primitive_variables;
  }

  template <int dim, typename number>
  const dealii::DoFHandler<dim> &
  DGCompressibleFlowOperation<dim, number>::get_dof_handler() const
  {
    return flow_scratch_data.scratch_data.get_dof_handler(flow_scratch_data.dof_idx);
  }

} // namespace MeltPoolDG::Flow
