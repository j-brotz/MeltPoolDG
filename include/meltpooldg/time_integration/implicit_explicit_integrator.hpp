#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_wrapper.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>

#include <meltpooldg/utilities/matrix_type_wrapper.h>

#include <functional>
#include <optional>

namespace MeltPoolDG::TimeIntegration
{
  /// The time integrator schemes supported by the implicit-explicit time integrator.
  inline static constexpr std::array<TimeIntegratorSchemes, 1> imex_supported_schemes{
    {TimeIntegratorSchemes::imex}};


  template <unsigned int dim, typename number>
  class ImplicitExplicitIntegrator final : public TimeIntegratorBase<number>
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    using ExplicitRhsFunctionType = std::function<void(number,
                                                       number,
                                                       VectorType &,
                                                       const VectorType &,
                                                       const bool,
                                                       std::function<void(unsigned, unsigned)>)>;

    using JacobianType =
      std::function<void(number time, number time_step, VectorType &dst, const VectorType &src)>;

    using ResidualType = std::function<void(number            time,
                                            number            time_step,
                                            const VectorType &src,
                                            VectorType       &dst,
                                            const VectorType &explicit_solution)>;

    using DistributeConstraintsType = std::function<void(VectorType &dst)>;

    using CustomSolverType = std::function<void(number            time,
                                                number            time_step,
                                                const VectorType &explicit_step_solution,
                                                const VectorType &solution)>;

  public:
    /**
     * Constructor. After construction it is still required to configure the explicit and implicit
     * steps using @ref configure_explicit_step() and @ref configure_implicit_step() and make a call to
     * @ref reinit() before the integrator can be used.
     *
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     */
    ImplicitExplicitIntegrator(const TimeIntegratorData<number> &time_integrator_data)
      : TimeIntegratorBase<number>(time_integrator_data)
    {}

    /**
     * Returns the number of previous solutions, that is solutions at time step n - x, where x >= 0,
     * required by the time integrator.
     */
    unsigned int
    required_solution_history_size() const override
    {
      return 2;
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
      intermediate_explicit_solution.reinit(vector_template, true);
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
     * @brief Configure the function used to compute the explicit right-hand side.
     *
     * Sets the class member @ref explicit_compute_rhs to the provided function.
     * For details on the expected function signature and behavior, see the
     * documentation of the corresponding class member.
     *
     * @param explicit_rhs Function to compute the right-hand side in the explicit step.
     */
    void
    configure_explicit_step(ExplicitRhsFunctionType explicit_rhs)
    {
      explicit_compute_rhs = explicit_rhs;
    }

    /**
     * Configure a custom solver function used to solve the implicit step.
     *
     * If this function is called the integrator will not use its internal nonlinear solver. For
     * details on the functions see the corresponding class member descriptions.
     *
     * Sets the class member @ref custom_solver to the provided function.
     * For details on the expected function signature and behavior, see the
     * documentation of the corresponding class member.
     *
     * @param custom_solver_in Custom solver function to be used in the implicit step.
     */
    void
    configure_implicit_step(CustomSolverType custom_solver_in)
    {
      custom_solver = custom_solver_in;
    }

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
     *
     * @note If a custom solver has already been set by calling @ref configure_implicit_step(CustomSolverType)
     * this function has no effect.
     */
    void
    configure_implicit_step(
      JacobianType              jacobian,
      ResidualType              residual,
      DistributeConstraintsType constraints = [](VectorType &) {})
    {
      compute_jacobian       = jacobian;
      compute_residual       = residual;
      distribute_constraints = constraints;

      solver.emplace(
        NewtonRaphsonSolver<number, VectorType>(this->time_integrator_data.nlsolver_data));

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
     *
     * @note The precondiitioner is only used for the default internal solver of the class. If a
     * custom solver function is set it is not used at all.
     */
    void
    set_preconditioner(Preconditioner<dim, VectorType, number> &&preconditioner_in)
    {
      preconditioner             = std::move(preconditioner_in);
      preconditioner_update_flag = true;
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
      SolutionHistory<VectorType>                                         &solution_history,
      const std::function<void(number, VectorType &, const VectorType &)> &stage_pre_processing,
      const std::function<void(number, VectorType &, const VectorType &)> &stage_post_processing)
      override
    {
      if (stage_pre_processing)
        stage_pre_processing(current_time,
                             solution_history.get_current_solution(),
                             solution_history.get_current_solution());

      apply_explicit_step(current_time,
                          time_step,
                          solution_history.get_current_solution(),
                          intermediate_explicit_solution);

      apply_implicit_step(current_time,
                          time_step,
                          intermediate_explicit_solution,
                          solution_history.get_current_solution());



      if (stage_post_processing)
        stage_post_processing(current_time,
                              solution_history.get_current_solution(),
                              solution_history.get_current_solution());

      ++n_steps_performed;
    }

  private:
    /// @brief Compute the negative residual for the implicit step of the time integrator.
    ///
    /// Assuming that the explicit part of the update
    /// \f[
    ///   \tilde{y} = y^{n} + \Delta t F(y^{n})
    /// \f]
    /// has already been computed and is provided as `explicit_step_solution`, this function
    /// evaluates the residual of the implicit equation
    /// \f[
    ///   R = \frac{y^{n+1} - \tilde{y}}{\Delta t} - G(y^{n+1}),
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
    ///        const VectorType &explicit_step_solution);
    ///```
    ///
    /// **Parameters:**
    /// - `time`                   : Current simulation time \f$t^n\f$.
    /// - `time_step`              : Current time step size \f$\Delta t\f$.
    /// - `dst`                    : Destination vector to store the negative residual.
    /// - `src`                    : Current solution vector, i.e., \f$ y^{n+1} \f$.
    /// - `explicit_step_solution` : Result \f$\tilde{y}\f$ from the explicit step.
    ///                              This vector may be modified safely, as it will be overwritten
    ///                              during the next integration step.
    ///
    /// @note This function is only used if the internal nonlinear solver of the class is used.
    ResidualType compute_residual;

    /// @brief Apply the Jacobian of the residual operator to a given vector.
    ///
    /// This function computes the action of the Jacobian, associated with the residual
    /// \f[
    ///   R = \frac{y^{n+1} - \tilde{y}}{\Delta t} - G(y^{n+1}),
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
    /// - `time_step` : Current time step size \f$\Delta t\f$.
    /// - `dst`       : Destination vector to store the Jacobian–vector product.
    /// - `src`       : Input vector to which the Jacobian is applied.
    ///
    /// @note This function is only used if the internal nonlinear solver of the class is used.
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
    ///
    /// @note This function is only used if the internal nonlinear solver of the class is used.
    DistributeConstraintsType distribute_constraints;

    //// Solve the implicit step of the implicit–explicit scheme.
    ///
    /// For an ODE of the form
    /// \f[
    ///   \dot{y} = F(y) + G(y),
    /// \f]
    /// where \f$F\f$ is treated explicitly and \f$G\f$ implicitly, this function advances the
    /// solution by one time step.
    ///
    /// Having already computed the explicit part
    /// \f[
    ///   \tilde{y} = y^{n} + \Delta t F(y^{n}),
    /// \f]
    /// the function solves the implicit equation
    /// \f[
    ///   y^{n+1} = \tilde{y} + \Delta t G(y^{n+1}),
    /// \f]
    /// to obtain the new solution \f$y^{n+1}\f$.
    ///
    /// **Function Signature:**
    /// ```cpp
    /// void f(number time,
    ///        number time_step,
    ///        VectorType &explicit_step_solution,
    ///        VectorType &solution);
    /// ```
    ///
    /// **Parameters:**
    /// - `time`                   : Current simulation time at \f$t^{n+1}\f$.
    /// - `time_step`              : Current step size \f$\Delta t\f$.
    /// - `explicit_step_solution` : Result \f$\tilde{y}\f$ from the explicit step. May be safely
    /// modified
    ///                              since it will be overwritten in the next step.
    /// - `solution`               : Solution vector \f$y^{n+1}\f$ at the new time step. On input,
    ///                              contains the previous solution \f$y^n\f$; on output, the
    ///                              updated solution \f$y^{n+1}\f$.
    CustomSolverType custom_solver;

    /// Explicit right-hand side function for the ODE system
    /// \f[
    ///   y' = F(y) + G(y),
    /// \f]
    /// where \f$F\f$ is treated explicitly and \f$G\f$ implicitly.
    /// This function computes the explicit part \f$F(y)\f$.
    ///
    /// **Function Signature:**
    /// ```cpp
    /// void f(number time,
    ///        number time_step,
    ///        VectorType &dst,
    ///        const VectorType &src,
    ///        const bool zero_dst_vec,
    ///        std::function<void(unsigned, unsigned)> post);
    /// ```
    ///
    /// **Parameters:**
    /// - `time`         : Current simulation time at \f$t^n\f$.
    /// - `time_step`    : Current step size \f$\Delta t\f$.
    /// - `dst`          : Destination vector for the explicit RHS contribution.
    /// - `src`          : Source solution vector \f$y^n\f$.
    /// - `zero_dst_vec` : If `true`, zero out `dst` before adding contributions. This allows
    ///                    efficient accumulation while computing values.
    /// - `post`         : Post-processing function applied after computing the RHS. Receives
    ///                    a range of global indices `[begin, end)` to process, ensuring all
    ///                    indices are handled. Designed for efficient integration with deal.II’s
    ///                    matrix-free framework.
    ExplicitRhsFunctionType explicit_compute_rhs;

    /// Vector to store the solution of the explicit step.
    VectorType intermediate_explicit_solution;

    /// Nonlinear solver used when no custom solver is provided.
    std::optional<NewtonRaphsonSolver<number, VectorType>> solver;

    /// Preconditioner for the linear solver used within each nonlinear solver iteration.
    Preconditioner<dim, VectorType, number> preconditioner;

    /// Boolean to indicate whether the preconditioner needs to be updated before the next solve.
    bool preconditioner_update_flag = false;

    /// Number of time steps already performed by the integrator.
    unsigned n_steps_performed = 0;

    /**
     * Apply the explicit and store the result in @p dst. The solution at \f$ t^n \f$ is given by the
     * vector @p src.
     */
    void
    apply_explicit_step(number time, number time_step, const VectorType &src, VectorType &dst) const
    {
      Assert(explicit_compute_rhs,
             dealii::ExcMessage(
               "No function has been set for computing the right-hand side in the explicit step!"));

      dst.zero_out_ghost_values();
      explicit_compute_rhs(
        time, time_step, dst, src, true, [&](const unsigned start_range, const unsigned end_range) {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (unsigned int i = start_range; i < end_range; ++i)
            {
              dst.local_element(i) *= time_step;
              dst.local_element(i) += src.local_element(i);
            }
        });
    }

    /**
     * Solve the implicit step of the implicit-explicit scheme. If the function @ref custom_solver
     * has been set this function is used, otherwise the default nonlinear solver of the class
     * is used. The result, i.e., the solution at the new time step, is stored in @p solution.
     * The solution after the explicit step is given by @p explicit_solution.
     */
    void
    apply_implicit_step(number      time,
                        number      time_step,
                        VectorType &explicit_solution,
                        VectorType &solution)
    {
      if (custom_solver)
        {
          custom_solver(time, time_step, explicit_solution, solution);
        }
      else
        {
          Assert(
            compute_residual,
            dealii::ExcMessage(
              "No function has been set for computing the computing the residual in the implicit step!"));
          Assert(
            compute_jacobian,
            dealii::ExcMessage(
              "No function has been set for computing the computing the jacobian in the implicit step!"));
          Assert(solver.has_value(), dealii::ExcInternalError());

          // update preconditioner if required
          if (n_steps_performed % this->time_integrator_data.preconditioner_update_frequency == 0 or
              preconditioner_update_flag)
            {
              preconditioner.update();
              preconditioner_update_flag = false;
            }
          solver->norm_of_solution_vector = [&solution = solution]() -> number {
            return solution.l2_norm();
          };

          // setup solver
          solver->distribute_constraints = [&](VectorType &solution) -> void {
            if (distribute_constraints)
              distribute_constraints(solution);
          };

          solver->reinit_vector = [&solution = solution](VectorType &vec) -> void {
            vec.reinit(solution);
          };

          solver->residual = [&](const VectorType &src, VectorType &dst) {
            compute_residual(time, time_step, src, dst, intermediate_explicit_solution);
          };

          std::function<void(VectorType &, const VectorType &)> jacobian_multiplication =
            [&](VectorType &dst, const VectorType &src) {
              compute_jacobian(time, time_step, dst, src);
            };

          MatrixTypeObject<VectorType> jacobian(jacobian_multiplication);

          solver->solve_with_jacobian = [&jacobian = jacobian,
                                         &data     = this->time_integrator_data.linear_solver_data,
                                         &preconditioner = preconditioner](const VectorType &rhs,
                                                                           VectorType       &dst) {
            return LinearSolver::solve(jacobian, dst, rhs, data, preconditioner);
          };

          // use explicit solution as initial guess
          solution = explicit_solution;
          solver->solve(solution);
        }
    }
  };
} // namespace MeltPoolDG::TimeIntegration