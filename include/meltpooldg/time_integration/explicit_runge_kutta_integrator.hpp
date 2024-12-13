/**
 * @brief Class providing different explicit (non low storage) Runge-Kutta time integrators.
 */


#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/time_integration/explicit_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace MeltPoolDG::TimeIntegration
{
  using namespace dealii;

  template <typename Number, typename PDEOperator = std::monostate>
  class ExplicitRungeKuttaIntegrator final : public ExplicitIntegratorBase<Number, PDEOperator>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    /**
     * Constructor. Set the coefficients for the explicit Runge-Kutta scheme.
     *
     *  @param parameters Integrator parameters.
     *  @param timer Timer used for computation time tracking.
     */
    ExplicitRungeKuttaIntegrator(const TimeIntegratorData &parameters, TimerOutput &timer)
      : parameters(parameters)
      , timer(timer)
    {
      if (parameters.integrator_type == TimeIntegratorSchemes::explicit_euler)
        {
          // Single-stage, first order Runge-Kutta scheme (explicit Euler)
          n_stages = 1;
          bi       = {{1.0}};
          ci       = {{0.0}};
        }
      else if (parameters.integrator_type == TimeIntegratorSchemes::RK_stage_2_order_2)
        {
          // Two-stage, second order Runge-Kutta scheme
          n_stages = 2;
          bi       = {{0.5, 0.5}};
          ci       = {{0.0, 1.0}};
          aij.resize(2);
          aij[1] = {{1.0}};
        }
      else if (parameters.integrator_type == TimeIntegratorSchemes::RK_stage_3_order_3)
        {
          // Three-stage, third order Runge-Kutta scheme
          n_stages = 3;
          bi       = {{1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0}};
          ci       = {{0.0, 0.5, 1.0}};
          aij.resize(3);
          aij[1] = {{0.5}};
          aij[2] = {{-1.0, 2.0}};
        }
      else if (parameters.integrator_type == TimeIntegratorSchemes::RK_stage_4_order_4)
        {
          // Four-stage, fourth order Runge-Kutta scheme
          n_stages = 4;
          bi       = {{1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0}};
          ci       = {{0.0, 0.5, 0.5, 1.0}};
          aij.resize(4);
          aij[1] = {{0.5}};
          aij[2] = {{0.0, 0.5}};
          aij[3] = {{0.0, 0.0, 1.0}};
        }
      else
        AssertThrow(false, dealii::ExcNotImplemented());
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
      vec_k.resize(n_stages);
      for (unsigned int i = 0; i < n_stages; ++i)
        vec_k[i].reinit(vector_template);
      stage_sum.reinit(vector_template);
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

    unsigned int
    required_solution_history_size() const override
    {
      return 1;
    }

    /**
     * Perform the actual time integration for a single time step.
     *
     * @param pde_operator Class providing the 'apply_operator()' function to compute f(y).
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     */
    void
    perform_time_step(const PDEOperator                              &pde_operator,
                      const double                                    current_time,
                      const double                                    time_step,
                      ::TimeIntegration::SolutionHistory<VectorType> &solution_history) override
    {
      AssertDimension(aij.size(), bi.size());
      TimerOutput::Scope timer_section(timer, "Explicit Runge-Kutta time integration");
      // compute the stages
      for (unsigned int stage = 0; stage < n_stages; ++stage)
        {
          compute_stage_sum(
            stage, time_step, vec_k, solution_history.get_current_solution(), stage_sum);

          pde_operator.apply_operator(current_time + ci[stage] * time_step,
                                      stage_sum,
                                      vec_k[stage],
                                      std::function<void(unsigned int, unsigned int)>());
        }

      // sum the stages
      for (unsigned int stage = 0; stage < n_stages; ++stage)
        {
          solution_history.get_current_solution().sadd(1., time_step * bi[stage], vec_k[stage]);
        }
    }

    /**
     * Computes the intermediate result of y used in the calculation of the stage's k vector.
     * The function returns the intermediate value y_int = y^(n) + dt * sum_{i=0}^s [a_ji * k_i],
     * where j represents the current stage (@p stage).
     *
     * @param stage The index of the current stage (starting from 0).
     * @param time_step The current time step size.
     * @param stages_k A collection of vectors k from the previous stages.
     * @param solution The solution at t^(n).
     * @param sum The destination vector where the summed stages are stored.
     */
    void
    compute_stage_sum(const unsigned int             stage,
                      const Number                   time_step,
                      const std::vector<VectorType> &stages_k,
                      const VectorType              &solution,
                      VectorType                    &sum) const
    {
      Assert(
        stages_k.size() >= stage,
        dealii::ExcMessage(
          "The number of computed previous stages must be the same as the number of the current stage."));
      sum = solution;
      for (unsigned int j = 0; j < stage; ++j)
        {
          // only add to the sum if coefficient a_ij is not zero
          if (aij[stage][j] != 0.)
            {
              sum.sadd(1, aij[stage][j] * time_step, stages_k[j]);
            }
        }
    }

    inline static constexpr std::array<TimeIntegratorSchemes, 4> supported_schemes{
      {TimeIntegratorSchemes::explicit_euler,
       TimeIntegratorSchemes::RK_stage_2_order_2,
       TimeIntegratorSchemes::RK_stage_3_order_3,
       TimeIntegratorSchemes::RK_stage_4_order_4}};

  private:
    unsigned int                     n_stages;
    std::vector<std::vector<double>> aij;
    std::vector<double>              bi;
    std::vector<double>              ci;

    std::vector<VectorType> vec_k;
    VectorType              stage_sum;

    const TimeIntegratorData parameters;
    TimerOutput             &timer;
  };

  // Define an alias for static access of variables
  using StaticExplicitRungeKuttaIntegrator = ExplicitRungeKuttaIntegrator<double>;

} // namespace MeltPoolDG::TimeIntegration