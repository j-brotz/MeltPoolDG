#pragma once



#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

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
     * Constructor. Sets up the nonlinear solver. After construction it is still required to set the required 
     * functions by calling @ref configure_solver_functions() and calling @ref reinit() to allocate required memory before the integrator can be used.
     *
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     */
    explicit BDFIntegrator(const TimeIntegratorData<number> &time_integrator_data);

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
      DistributeConstraintsType constraints = [](VectorType &) {});

    /**
     * Set the preconditioner used in the linear solver of the implicit step. If this function is
     * never called an identity preconditioner is used.
     *
     * @param preconditioner_in Preconditioner to be used in the linear solver of the implicit step.
     */
    void
    set_preconditioner(Preconditioner<dim, VectorType, number> &&preconditioner_in);

    /**
     * Returns the number of previous solutions, that is solutions at time step n - x, where x >= 0,
     * required by the time integrator.
     */
    unsigned int
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
      override;

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
    compute_weighted_old_solution_sum(SolutionHistory<VectorType> &solution_history);

    /**
     * @brief Set the up bdf parameters object.
     *
     * This function internally sets the bdf weights and coefficients used in the time integration
     * scheme.
     *
     * @param bdf_scheme Number of tbe bdf scheme to be used.
     */
    void
    set_up_bdf_parameters(const unsigned int bdf_scheme);
  };
} // namespace MeltPoolDG::TimeIntegration