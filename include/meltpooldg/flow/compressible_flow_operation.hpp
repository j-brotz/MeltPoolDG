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
#include <meltpooldg/flow/compressible_flow_operator_explicit.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/explicit_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

#include <memory>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim, typename number = double>
  class CompressibleFlowOperation
  {
    using VectorType = LinearAlgebra::distributed::Vector<number>;

  public:
    /**
     * Constructor.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param comp_flow_data_in Reference to the compressible flow data struct used.
     * @param time_integrator_data_in Reference to the used time integrator data.
     * @param comp_flow_dof_idx_in Index of the used dof handler in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     */
    CompressibleFlowOperation(const ScratchData<dim>                    &scratch_data_in,
                              const CompressibleFlowData                &comp_flow_data_in,
                              const TimeIntegration::TimeIntegratorData &time_integrator_data_in,
                              unsigned int                               comp_flow_dof_idx_in  = 0,
                              unsigned int                               comp_flow_quad_idx_in = 0);

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
    solve(double current_time, double time_step);

    /**
     * Set an inflow boundary conditions for all boundary ids occurring in the given std::map and
     * applies the corresponding function at this boundary. This corresponds to a Dirichlet boundary
     * condition to all primary variables.
     *
     * @param inflow_bc Map of boundary ids and corresponding functions for inflow boundaries.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_inflow_boundary(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &inflow_bc);

    /**
     * Set a subsonic outflow boundary condition for all boundary ids occurring in the given
     * std::map and applies the corresponding function at this boundary. This corresponds to a
     * Dirichlet boundary condition for the static pressure part of the energy. The dynamic part is
     * added according to the locally present velocity values as it is proposed in Hartmann R. and
     * Houston P., An Optimal Order Interior Penalty Discontinuous Galerkin Discretization of the
     * Compressible Navier–Stokes Equations.
     *
     * @param outflow_fixed_pressure_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_subsonic_outflow_with_fixed_static_pressure(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
        &outflow_fixed_pressure_bc);

    /**
     * Set a subsonic outflow boundary condition with prescribed energy for all boundary ids
     * occurring in the given std::map and applies the corresponding function at this boundary. This
     * corresponds to a Dirichlet boundary condition for the energy.
     *
     * @param outflow_fixed_energy_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_subsonic_outflow_with_fixed_energy(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &outflow_fixed_energy_bc);

    /**
     * Set a slip wall boundary condition for all boundary ids occurring in the given std::map,
     * where the corresponding functions are not used for the boundary condition. This represents a
     * symmetry condition for the momentum (normal velocity results to zero).
     *
     * @param slip_wall_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_slip_wall_boundary(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &slip_wall_bc);

    /**
     * Set an adiabatic no-slip wall boundary condition for all boundary ids occurring in the given
     * std::map, where the corresponding functions are not used for the boundary condition. This
     * represents a homogeneous Dirichlet boundary condition for the momentum and a homogeneous
     * Neumann boundary condition for the energy.
     *
     * @param no_slip_wall_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_no_slip_adiabatic_wall_boundary(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &no_slip_wall_bc);

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

    void
    set_initial_condition(const Function<dim> &function);

    /**
     * Attach the solution to the passed data out object. The solution which are added are the
     * density, the momentum and the energy density.
     *
     * @param data_out Object to which the solution vector is attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    /**
     * Getter functions.
     */
    const VectorType &
    get_solution() const;

    VectorType &
    get_solution();


  private:
    ::TimeIntegration::SolutionHistory<VectorType> solution_history_;

    const ScratchData<dim>    &scratch_data_;
    const CompressibleFlowData comp_flow_data_;

    const unsigned int comp_flow_dof_idx  = 0;
    const unsigned int comp_flow_quad_idx = 0;

    std::unique_ptr<CompressibleFlowOperatorExplicit<dim, number>> comp_flow_operator_;

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

    std::unique_ptr<TimeIntegration::
                      ExplicitIntegratorBase<number, CompressibleFlowOperatorExplicit<dim, number>>>
      time_integrator;
  };


  //! inlined functions
  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  CompressibleFlowOperation<dim, number>::get_solution() const
  {
    return solution_history_.get_current_solution();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  CompressibleFlowOperation<dim, number>::get_solution()
  {
    return solution_history_.get_current_solution();
  }

} // namespace MeltPoolDG::Flow