#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <meltpooldg/utilities/matrix_type_wrapper.h>

#include <functional>
#include <memory>
#include <vector>

namespace MeltPoolDG::TimeIntegration
{
  /// The time integrator schemes supported by the bdf time integrator.
  inline static constexpr std::array<TimeIntegratorSchemes, 6> bdf_supported_schemes{
    {TimeIntegratorSchemes::bdf_1,
     TimeIntegratorSchemes::bdf_2,
     TimeIntegratorSchemes::bdf_3,
     TimeIntegratorSchemes::bdf_4,
     TimeIntegratorSchemes::bdf_5,
     TimeIntegratorSchemes::bdf_6}};

  template <int dim, typename number>
  class BDFIntegrator final : public TimeIntegratorBase<number>
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    using JacobianType =
      std::function<void(number time_step, VectorType &src, const VectorType &dst)>;

    using ResidualType = std::function<void(number            time,
                                            number            time_step,
                                            const VectorType &src,
                                            VectorType       &dst,
                                            const VectorType &old_solution)>;

    using DistributeConstraintsType = std::function<void(VectorType &dst)>;

    /**
     * Constructor. Sets up the nonlinear solver. After construction it is still required to set the required functions by calling @ref configure_solver_functions() and calling @ref reinit() to allocate reuired memory before the integrator can be used.
     *
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     */
    BDFIntegrator(const TimeIntegratorData<number> &time_integrator_data)
      : TimeIntegratorBase<number>(time_integrator_data)
      , solver(std::make_unique<NewtonRaphsonSolver<number, VectorType>>(
          this->time_integrator_data.nlsolver_data))
    {}

    /**
     * @brief Configure the functions used by the internal nonlinear solver to solve the implicit step. For
     * details on the functions see the corresponding class member descriptions.
     *
     * Sets the class member @ref compute_jacobian , @ref compute_residual and @ref distribute_constraints to the
     * provided functions. For details on the expected function signatures and behavior, see the
     * documentation of the corresponding class member.
     *
     * @param jacobian Function used to apply the Jacobian to a vector.
     * @param residual Function used to compute the residual.
     * @param constraints Function used to apply constraints to a vector.
     */
    void
    configure_solver_functions(
      JacobianType              jacobian,
      ResidualType              residual,
      DistributeConstraintsType constraints = [](VectorType &) {})
    {
      compute_jacobian       = jacobian;
      compute_residual       = residual;
      distribute_constraints = constraints;

      // Ensure that a preconditioner is set. If not set previously it is set to identity but can be
      // changed by the user anytime by calling set_preconditioner().
      if (!preconditioner.is_initialized())
        {
          preconditioner = Preconditioner<dim, VectorType, number>(
            IdentityPreconditioner<dim, VectorType, number>());
        }
      preconditioner_update_flag = true;
    }

    /**
     * Set the preconditioner used in the linear solver of the implicit step. If this function is
     * never called an identity preconditioner is used.
     *
     * @param preconditioner_in Preconditioner to be used in the linear solver of the implicit step.
     */
    void
    set_preconditioner(Preconditioner<dim, VectorType, number> &&preconditioner_in)
    {
      preconditioner             = std::move(preconditioner_in);
      preconditioner_update_flag = true;
    }

    /**
     * Returns the number of previous solutions, that is solutions at time step n - x, where x >= 0,
     * required by the time integrator.
     */
    unsigned int
    required_solution_history_size() const override
    {
      switch (this->time_integrator_data.integrator_type)
        {
          case TimeIntegratorSchemes::bdf_1:
            return 2;
          case TimeIntegratorSchemes::bdf_2:
            return 3;
          case TimeIntegratorSchemes::bdf_3:
            return 4;
          case TimeIntegratorSchemes::bdf_4:
            return 5;
          case TimeIntegratorSchemes::bdf_5:
            return 6;
          case TimeIntegratorSchemes::bdf_6:
            return 7;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    /**
     * Allocate memory for the required vectors used during the integration. This function needs to
     * be called once before the function @ref perform_time_step() can be called.
     *
     * @param vector_template Reference vector used to define the partitioning for all internal
     * vectors.
     */
    void
    reinit(const VectorType &vector_template) override
    {
      summed_old_solution.reinit(vector_template, true);
      preconditioner_update_flag = true;
    }

    /**
     * Sets up the necessary internal data structures by internally calling
     * @ref reinit(solution_history.get_current_solution()).
     */
    void
    reinit(const SolutionHistory<VectorType> &solution_history) override
    {
      reinit(solution_history.get_current_solution());
    }

    /**
     * Perform the actual time integration for a single time step using the low storage explicit
     * Runge-Kutta scheme.
     *
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     * @param stage_pre_processing Function which is executed before the bdf update step.
     * @param stage_post_processing Function which is executed after the bdf update step.
     * Runge-Kutta stage.
     */
    void
    perform_time_step(
      const number                                                         current_time,
      const number                                                         time_step,
      SolutionHistory<VectorType>                                         &solution_history,
      const std::function<void(number, VectorType &, const VectorType &)> &stage_pre_processing,
      const std::function<void(number, VectorType &, const VectorType &)> &stage_post_processing)
      override
    {
      Assert(compute_jacobian and compute_residual,
             dealii::ExcMessage("The integrator has not been initialized!"));

      // Perform the initial steps with a bdf scheme of lower order, i.e. 1st step: BDF1, 2nd
      // step BDF2 and so on until the desired BDF scheme is reached
      static unsigned step_count = 0;
      if (step_count != required_solution_history_size() - 1)
        {
          ++step_count;
          set_up_bdf_parameters(step_count);
        }

      Assert(solution_history.size() > step_count,
             dealii::ExcMessage(
               "The size of the solution history object does not fit the requirements of the "
               "chosen time integration scheme."));

      compute_weighted_old_solution_sum(solution_history);

      if (stage_pre_processing)
        stage_pre_processing(current_time,
                             solution_history.get_current_solution(),
                             solution_history.get_current_solution());

      if (n_steps_performed % this->time_integrator_data.preconditioner_update_frequency == 0 or
          preconditioner_update_flag)
        {
          preconditioner.update();
          preconditioner_update_flag = false;
        }

      // matrix type wrapper for the jacobian
      std::function<void(VectorType &, const VectorType &)> jacobian_multiplication =
        [&](VectorType &dst, const VectorType &src) {
          // We pass time_step*bdf_weights.rhs as time step as the right hand side f(y) is not
          // scaled and therefore the complete equation is divided by this factor.
          compute_jacobian(time_step * bdf_weights.rhs, dst, src);
        };

      MatrixTypeObject<VectorType> jacobian(jacobian_multiplication);

      solver->norm_of_solution_vector = [&solution =
                                           solution_history.get_current_solution()]() -> number {
        return solution.l2_norm();
      };

      solver->distribute_constraints = [&](VectorType &solution) -> void {
        if (distribute_constraints)
          distribute_constraints(solution);
      };

      solver->reinit_vector = [&solution = solution_history.get_current_solution()](
                                VectorType &vec) -> void { vec.reinit(solution); };

      solver->residual = [&](const VectorType &src, VectorType &dst) {
        // We pass time_step*bdf_weights.rhs as time step as the right hand side f(y) is not scaled
        // and therefore the complete equation is divided by this factor.
        compute_residual(
          current_time + time_step, time_step * bdf_weights.rhs, src, dst, summed_old_solution);
      };

      solver->solve_with_jacobian = [&jacobian = jacobian,
                                     &data     = this->time_integrator_data.linear_solver_data,
                                     &preconditioner = preconditioner](const VectorType &rhs,
                                                                       VectorType       &dst) {
        return LinearSolver::solve(jacobian, dst, rhs, data, preconditioner);
      };

      solver->solve(solution_history.get_current_solution());

      if (stage_post_processing)
        stage_post_processing(current_time + time_step,
                              solution_history.get_current_solution(),
                              solution_history.get_current_solution());

      ++n_steps_performed;
    }

  private:
    /// @brief Compute the negative residual for the implicit step of the time integrator.
    ///
    /// This function evaluates the residual given by
    /// \f[
    ///   R = \frac{y^{n+1} - y^{n}}{\Delta t} - F(y^{n+1}),
    /// \f]
    /// and returns its negative, i.e., \f$ -R \f$.
    /// The negative residual is required directly by the internal nonlinear solver.
    ///
    /// **Function Signature:**
    ///```cpp
    /// void f(number time,
    ///        number time_step,
    ///        VectorType &dst,
    ///        const VectorType &src,
    ///        const VectorType &old_solution);
    ///```
    ///
    /// **Parameters:**
    /// - `time`         : Current simulation time \f$t^n\f$.
    /// - `time_step`    : Current time step size \f$\Delta t\f$ (modified by the BDF prefactor).
    /// - `dst`          : Destination vector to store the negative residual.
    /// - `src`          : Current solution vector, i.e., \f$ y^{n+1} \f$.
    /// - `old_solution` : Previous time step resut, i.e., \f$ y^n \f$ in the above formula. in fact
    /// this is a modification of the previous solution including all previous solutions scaled by
    /// their weights for the corresponding BDF scheme.
    ResidualType compute_residual;

    /// @brief Apply the Jacobian of the residual operator to a given vector.
    ///
    /// This function computes the action of the Jacobian, associated with the residual
    /// \f[
    ///  R = \frac{y^{n+1} - y^{n}}{\Delta t} - F(y^{n+1}),
    /// \f]
    /// on the input vector `src` and stores the result in `dst`.
    ///
    /// **Function Signature:**
    /// ```cpp
    /// void f(number time,
    ///        number time_step,
    ///        VectorType &dst,
    ///        const VectorType &src);
    /// ```
    ///
    /// **Parameters:**
    /// - `time`      : Current simulation time \f$t^n\f$.
    /// - `time_step` : Current time step size \f$\Delta t\f$ (modified by the BDF prefactor).
    /// - `dst`       : Destination vector to store the Jacobian–vector product.
    /// - `src`       : Input vector to which the Jacobian is applied.
    JacobianType compute_jacobian;

    /// Apply constraints to a given vector.
    ///
    /// This function enforces problem-specific constraints (e.g., boundary conditions
    /// or degrees of freedom restrictions) directly on the input vector `dst`.
    ///
    /// **Function Signature:**
    /// ```cpp
    /// void f(VectorType &dst);
    /// ```
    ///
    /// **Parameters:**
    /// - `dst` : Vector to which the constraints are applied (modified in place).
    DistributeConstraintsType distribute_constraints;

    /// Nonlinear solver
    std::unique_ptr<NewtonRaphsonSolver<number, VectorType>> solver;

    /// Preconditioner for the linear solver used in each nonlinear iteration.
    Preconditioner<dim, VectorType, number> preconditioner;

    /// BDF prefactors and weights.
    struct
    {
      /// Prefactor for the right hand side f(y).
      number rhs;
      /// Weights for the old solutions y^{n}, y^{n-1}, ...
      std::vector<number> old_solutions;
    } bdf_weights;

    /// Sum of old solution with prefactors from BDF method.
    VectorType summed_old_solution;

    /// Boolean to indicate whether the preconditioner needs to be updated before the next solve.
    bool preconditioner_update_flag = false;

    /// Number of time steps already performed by the integrator.
    unsigned n_steps_performed = 0;

    /**
     * @brief Compute sum of old solutions weighted with the BDF prefactors @ref bdf_weights.old_solutions .
     *
     * @param solution_history History object containing the required old solutions.
     */
    void
    compute_weighted_old_solution_sum(SolutionHistory<VectorType> &solution_history)
    {
      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < summed_old_solution.locally_owned_size(); ++i)
        {
          summed_old_solution.local_element(i) = 0.;
          for (unsigned int j = bdf_weights.old_solutions.size(); j > 0; --j)
            {
              summed_old_solution.local_element(i) +=
                bdf_weights.old_solutions[j - 1] *
                solution_history.get_solution(j).local_element(i);
            }
        }
    }

    /**
     * @brief Set the up bdf parameters object.
     *
     * This function internally sets the bdf weights and coefficients used in the time integration
     * scheme.
     *
     * @param bdf_scheme Number of tbe bdf scheme to be used.
     */
    void
    set_up_bdf_parameters(const unsigned int bdf_scheme)
    {
      bdf_weights.old_solutions.clear();
      switch (bdf_scheme)
        {
            case 1: {
              bdf_weights.rhs = 1.;
              bdf_weights.old_solutions.reserve(1);
              bdf_weights.old_solutions.push_back(1.);
              break;
            }
            case 2: {
              bdf_weights.rhs = 2. / 3.;
              bdf_weights.old_solutions.reserve(2);
              bdf_weights.old_solutions.push_back(4. / 3.);
              bdf_weights.old_solutions.push_back(-1. / 3.);
              break;
            }
            case 3: {
              bdf_weights.rhs = 6. / 11.;
              bdf_weights.old_solutions.reserve(3);
              bdf_weights.old_solutions.push_back(18. / 11.);
              bdf_weights.old_solutions.push_back(-9. / 11.);
              bdf_weights.old_solutions.push_back(2. / 11.);
              break;
            }
            case 4: {
              bdf_weights.rhs = 12. / 25.;
              bdf_weights.old_solutions.reserve(4);
              bdf_weights.old_solutions.push_back(48. / 25.);
              bdf_weights.old_solutions.push_back(-36. / 25.);
              bdf_weights.old_solutions.push_back(16. / 25.);
              bdf_weights.old_solutions.push_back(-3. / 25.);
              break;
            }
            case 5: {
              bdf_weights.rhs = 60. / 137.;
              bdf_weights.old_solutions.reserve(5);
              bdf_weights.old_solutions.push_back(300. / 137.);
              bdf_weights.old_solutions.push_back(-300. / 137.);
              bdf_weights.old_solutions.push_back(200. / 137.);
              bdf_weights.old_solutions.push_back(-75. / 137.);
              bdf_weights.old_solutions.push_back(12. / 137.);
              break;
            }
            case 6: {
              bdf_weights.rhs = 60. / 147.;
              bdf_weights.old_solutions.reserve(6);
              bdf_weights.old_solutions.push_back(360. / 147.);
              bdf_weights.old_solutions.push_back(-450. / 147.);
              bdf_weights.old_solutions.push_back(400. / 147.);
              bdf_weights.old_solutions.push_back(-225. / 147.);
              bdf_weights.old_solutions.push_back(72. / 147.);
              bdf_weights.old_solutions.push_back(-10. / 147.);
              break;
            }
          default:
            Assert(false, dealii::ExcMessage("This code should not be reachable!"));
        }
    }
  };
} // namespace MeltPoolDG::TimeIntegration