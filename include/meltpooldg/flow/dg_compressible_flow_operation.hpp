/**
 * @brief This operator solves the compressible Navier-Stokes equations, comprising
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

#include <deal.II/lac/la_parallel_vector.h>

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
  template <int dim, typename number>
  class DGCompressibleFlowOperation
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    /**
     * Constructor.
     *
     * @param scratch_data Reference to the used ScratchData object.
     * @param flow_data Reference to the compressible flow data struct used.
     * @param material_data_in Reference to the material data struct.
     * @param flow_dof_idx Index of the used dof handler in @p scratch_data_in.
     * @param flow_quad_idx Index of the used quadrature object in @p scratch_data_in.
     * @param external_forces Pointer to a struct implementing external forces acting on the fluid.
     */
    DGCompressibleFlowOperation(
      const ScratchData<dim, dim, number>              &scratch_data,
      const CompressibleFlowData<number>               &flow_data,
      const CompressibleFluidMaterialPhaseData<number> &material_data_in,
      unsigned int                                      flow_dof_idx  = 0,
      unsigned int                                      flow_quad_idx = 0,
      std::unique_ptr<ExternalFluidForcesRightHandSideContribution<dim, number>> &&external_forces =
        nullptr);

    /**
     * Set up the required internal data structures. After a call to this function the solve()
     * function of the class can be utilized.
     */
    void
    reinit();

    /**
     * Solves the compressible Navier-Stokes equations for a single time step.
     *
     * @param current_time Current time at t^n.
     * @param time_step Current time step size.
     */
    void
    solve(const number current_time, const number time_step);

    /**
     * Distribute the degrees of freedom to the passed dof handler object.
     *
     * @param dof_handler Dof handler object ussed for the compressible flow solver.
     */
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const;

    /**
     * Set the boundary conditions.
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
     * Set a body force, e.g. gravity, specified by the passed function.
     *
     * @param body_force_in Function specifying the body force.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_body_force(std::unique_ptr<dealii::Function<dim>> body_force_in);

    /**
     * Compute the maximum time step size arising from the convective and viscous time step limits
     * and optionally print it to the console.
     *
     * @param do_print If true, the time step limit is printed to the console.
     */
    number
    compute_time_step_size(bool do_print = false) const;

    /**
     * Set the solution vector to the passed initial flow field state.
     *
     * @param function Initial condition of the flow field.
     */
    void
    set_initial_condition(const dealii::Function<dim> &function);

    /**
     * Attach the solution to the passed data out object. The solution which are added are the
     * density, the momentum and the energy density.
     *
     * @param data_out Object to which the solution vector is attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /**
     * Getter functions.
     */
    const VectorType &
    get_solution() const;

    VectorType &
    get_solution();

    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

  private:
    CompressibleFlowScratchData<dim, number> flow_scratch_data;

    std::unique_ptr<DGCompressibleFlowOperatorBase<number>> comp_flow_operator;

    std::unique_ptr<TimeIntegration::TimeIntegratorBase<number>> time_integrator;

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
     * Set up the operator to suit the specified time integration scheme.
     *
     * @param external_forces Pointer to a struct implementing external forces acting on the fluid.
     */
    void
    setup_operator_and_time_integrator(
      std::unique_ptr<ExternalFluidForcesRightHandSideContribution<dim, number>> &&external_forces);
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
  const dealii::DoFHandler<dim> &
  DGCompressibleFlowOperation<dim, number>::get_dof_handler() const
  {
    return flow_scratch_data.scratch_data.get_dof_handler(flow_scratch_data.dof_idx);
  }

} // namespace MeltPoolDG::Flow
