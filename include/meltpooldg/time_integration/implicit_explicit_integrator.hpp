#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_wrapper.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <functional>

namespace MeltPoolDG::TimeIntegration
{
  /// The time integrator schemes supported by the implicit-explicit time integrator.
  inline static constexpr std::array<TimeIntegratorSchemes, 1> imex_supported_schemes{
    {TimeIntegratorSchemes::imex}};


  template <unsigned int dim, typename number>
  class ImplicitExplicitIntegrator final : public TimeIntegratorBase<number>
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    using ExplicitRhsFunctionType =
      std::function<void(number,
                         number,
                         VectorType &,
                         const VectorType &,
                         const bool,
                         const std::function<void(unsigned, unsigned)> &)>;

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

    /**
     * Structure containing the functions used by the internal solver to solve the explicit and the
     * implicit step.
     */
    struct SolverFunctions
    {
      /// Function that applies the Jacobian of the residual operator of the implicit step to a
      /// given vector.
      JacobianType compute_jacobian;

      /// Function that computes the negative residual for the implicit step of the time integrator.
      ResidualType compute_residual;

      /// An optional function that applies constraints to a given vector. If not provided, no
      /// constraints will be applied.
      DistributeConstraintsType distribute_constraints;

      /// For the explicit step, this function computes the right-hand side of the ODE system. It is
      /// used to compute the explicit part of the update.
      ExplicitRhsFunctionType compute_explicit_rhs;
    };

    /**
     * Constructor. After construction it is still make a call to reinit() before the integrator can
     * be used.
     *
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     * @param solver_functions Struct containing the functions used by the internal solver to solve
     * the explicit and implicit step.
     * @param preconditioner_in Preconditioner to be used in the linear solver of the implicit step.
     */
    explicit ImplicitExplicitIntegrator(
      const TimeIntegratorData<number>         &time_integrator_data,
      const SolverFunctions                     solver_functions,
      Preconditioner<dim, VectorType, number> &&preconditioner_in =
        Preconditioner<dim, VectorType, number>(IdentityPreconditioner<dim, VectorType, number>()));

    /**
     * Returns the number of previous solutions, that is solutions at time step n - x, where x >= 0,
     * required by the time integrator.
     */
    unsigned
    required_solution_history_size() const override;

    /**
     * Allocate memory for the required vectors used during the integration. This function needs to
     * be called once before the function @ref perform_time_step() can be called.
     *
     * @param vector_template Reference vector used to define the partitioning for all internal
     * vectors.
     */
    void
    reinit(const VectorType &vector_template) override;

    /**
     * Sets up the necessary internal data structures by internally calling
     * @ref reinit(solution_history.get_current_solution()).
     */
    void
    reinit(const SolutionHistory<VectorType> &solution_history) override;

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
    perform_time_step(const number                 current_time,
                      const number                 time_step,
                      SolutionHistory<VectorType> &solution_history,
                      const std::function<void(number, number, VectorType &, const VectorType &)>
                        &stage_pre_processing,
                      const std::function<void(number, number, VectorType &, const VectorType &)>
                        &stage_post_processing) override;

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
    NewtonRaphsonSolver<number, VectorType> solver;

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
    apply_explicit_step(number            time,
                        number            time_step,
                        const VectorType &src,
                        VectorType       &dst) const;

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
                        VectorType &solution);
  };
} // namespace MeltPoolDG::TimeIntegration
