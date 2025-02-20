/**
 * @brief
 */

#pragma once

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>

#include <meltpooldg/utilities/matrix_type_wrapper.h>

namespace MeltPoolDG
{
  /**
   * The time integrator schemes supported by the bdf time integrator.
   */
  inline static constexpr std::array<TimeIntegratorSchemes, 1> imex_supported_schemes{
    {TimeIntegratorSchemes::imex}};


  template <unsigned int dim, typename number, typename PDEOperator>
  class ImplicitExplicitIntegrator final : public TimeIntegratorBase<number>
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    /**
     * Constructor. Set the coefficients for the BDF time integration scheme.
     *
     * @param pde_operator Operator providing the necessary functionality to perform the time
     * integration.
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     * @param scratch_data_in Scratch data object used in the pde operator.
     * @param dof_idx_in Relevant dof index of the passed operator in the scratch data object.
     */
    ImplicitExplicitIntegrator(const PDEOperator        &pde_operator,
                               const TimeIntegratorData &time_integrator_data,
                               const ScratchData<dim>   &scratch_data_in,
                               const unsigned int        dof_idx_in)
      : TimeIntegratorBase<number>(time_integrator_data)
      , pde_operator(pde_operator)
      , nonlinear_solver(this->time_integrator_data.nlsolver_data)
      , scratch_data(scratch_data_in)
      , dof_idx(dof_idx_in)
    {
      preconditioner = make_preconditioner<dim, PDEOperator, VectorType>(
        time_integrator_data.linear_solver_data.preconditioner_type, &pde_operator);
    }

    unsigned int
    required_solution_history_size() const override
    {
      return 2;
    }

    /**
     * Allocate memory for the required vectors used during the integration. This function needs to
     * be called once before the function perform_time_step() can be called.
     *
     * @param vector_template Reference vector used to define the partitioning for all internal
     * vectors.
     */
    void
    reinit(const VectorType &vector_template) override
    {
      intermediate_explicit_solution.reinit(vector_template, true);
      preconditioner.reinit(scratch_data, dof_idx);
    }

    /**
     * Sets up the necessary internal data structures by internally calling
     * reinit(solution_history.get_current_solution()).
     */
    void
    reinit(const ::TimeIntegration::SolutionHistory<VectorType> &solution_history) override
    {
      reinit(solution_history.get_current_solution());
    }

    /**
     * Perform a single time step by first computing an intermediate explicit solution using an
     * explicit Euler step for the part of the pde which is treated explicitly. The explicit step is
     * followed by an implicit Euler step for the implicit part of the PDE resulting in the solution
     * at the new time step.
     *
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     * @param stage_pre_processing Function which is executed before the explicit step.
     * @param stage_post_processing Function which is executed after the solution at the new time
     * step has been computed.
     */
    void
    perform_time_step(
      const number                                                         current_time,
      const number                                                         time_step,
      ::TimeIntegration::SolutionHistory<VectorType>                      &solution_history,
      const std::function<void(number, VectorType &, const VectorType &)> &stage_pre_processing,
      const std::function<void(number, VectorType &, const VectorType &)> &stage_post_processing)
      override
    {
      if (stage_pre_processing)
        stage_pre_processing(current_time,
                             solution_history.get_current_solution(),
                             solution_history.get_current_solution());

      pde_operator.set_stage_constants(current_time, time_step, intermediate_explicit_solution);

      pde_operator.perform_explicit_stage(current_time,
                                          time_step,
                                          intermediate_explicit_solution,
                                          solution_history.get_current_solution(),
                                          true);

      // matrix type wrapper for the jacobian
      std::function<void(VectorType &, const VectorType &)> jacobian_multiplication =
        MELT_POOL_DG_LAMBDA_WRAPPER(pde_operator.apply_jacobian);
      MatrixTypeObject<VectorType> jacobian(jacobian_multiplication);

      nonlinear_solver.norm_of_solution_vector =
        [&solution = solution_history.get_current_solution()]() -> double {
        return solution.l2_norm();
      };

      nonlinear_solver.distribute_constraints = [](VectorType &) -> void {};

      nonlinear_solver.reinit_vector = [&solution = solution_history.get_current_solution()](
                                         VectorType &vec) -> void { vec.reinit(solution); };

      nonlinear_solver.residual =
        [&current_time, &pde_operator = pde_operator](const VectorType &src, VectorType &dst) {
          pde_operator.compute_residual(current_time, src, dst);
        };

      nonlinear_solver.solve_with_jacobian =
        [&jacobian       = jacobian,
         &data           = this->time_integrator_data.linear_solver_data,
         &preconditioner = preconditioner](const VectorType &rhs, VectorType &dst) {
          return LinearSolver::solve(jacobian, dst, rhs, data, preconditioner);
        };

      solution_history.get_current_solution().zero_out_ghost_values();
      pde_operator.make_initial_guess(solution_history.get_current_solution());
      solution_history.get_current_solution().update_ghost_values();
      // TODO: We need a rule that determines when to update the preconditioner
      static int count = 0;
      if (count % this->time_integrator_data.preconditioner_update_frequency == 0)
        preconditioner.update();
      count++;
      nonlinear_solver.solve(solution_history.get_current_solution());

      if (stage_post_processing)
        stage_pre_processing(current_time,
                             solution_history.get_current_solution(),
                             solution_history.get_current_solution());
    }

  private:
    const PDEOperator &pde_operator;

    //! Preconditioner for the linear solver
    Preconditioner<dim, VectorType> preconditioner;

    //! Nonlinear solver
    NewtonRaphsonSolver<VectorType> nonlinear_solver;

    VectorType intermediate_explicit_solution;

    const ScratchData<dim> &scratch_data;

    const unsigned int dof_idx;
  };
} // namespace MeltPoolDG