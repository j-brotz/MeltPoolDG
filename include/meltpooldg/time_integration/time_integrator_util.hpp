
#pragma once

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/time_integration/explicit_low_storage_runge_kutta_integrator.hpp>
#include <meltpooldg/time_integration/one_step_theta.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

namespace MeltPoolDG
{
  /**
   * Checks if the given @p scheme is explicit and supported by one of the available
   * explicit integrator classes.
   *
   * @param scheme The time integration scheme to check.
   *
   * @return True if the scheme is explicit and supported; otherwise, false.
   */
  inline bool
  time_integrator_scheme_is_explicit(const TimeIntegratorSchemes scheme)
  {
    if (Utils::contains(explicit_lsrk_supported_schemes, scheme))
      return true;
    return false;
  }

  /**
   * Factory function that creates and returns a raw pointer to an explicit time integrator
   * based on the scheme specified in @p TimeIntegratorData.
   *
   * @param params Contains the configuration details for the time integrator.
   * @param timer Timer passed to the constructor of the time integrator.
   *
   * @return A raw pointer to the appropriate explicit time integrator.
   * @throws An exception if the specified integration scheme is not supported.
   * @note This function returns a raw pointer, leaving the responsibility for memory management
   * (e.g., wrapping it in a smart pointer) to the caller.
   */
  template <typename number, typename PDEOperator>
  TimeIntegratorBase<number, PDEOperator> *
  explicit_time_integrator_factory(const TimeIntegratorData &params, dealii::TimerOutput &timer)
  {
    if constexpr (ExplicitPDEOperator<PDEOperator,
                                      number,
                                      dealii::LinearAlgebra::distributed::Vector<number>>)
      {
        if (Utils::contains(explicit_lsrk_supported_schemes, params.integrator_type))
          return new LowStorageExplicitRungeKuttaIntegrator<number, PDEOperator>(params, timer);
      }
    DEAL_II_NOT_IMPLEMENTED();
  }


  /**
   * Factory function that creates and returns a raw pointer to any suitable time integrator
   * derived form the time integrator base class based on the scheme specified in
   * @p TimeIntegratorData.
   *
   * @param params Contains the configuration details for the time integrator.
   * @param linear_solver_data Data for the linear solver used in (semi-) implicit time stepping.
   * @param timer Timer passed to the constructor of the time integrator.
   *
   * @return A raw pointer to the appropriate time integrator.
   * @throws An exception if the specified integration scheme is not supported.
   * @note This function returns a raw pointer, leaving the responsibility for memory management
   * (e.g., wrapping it in a smart pointer) to the caller.
   */
  template <typename number, typename PDEOperator>
  TimeIntegratorBase<number, PDEOperator> *
  time_integrator_factory(const TimeIntegratorData       &params,
                          const LinearSolverData<number> &linear_solver_data,
                          dealii::TimerOutput            &timer)
  {
    if constexpr (ExplicitPDEOperator<PDEOperator,
                                      number,
                                      dealii::LinearAlgebra::distributed::Vector<number>>)
      {
        if (Utils::contains(explicit_lsrk_supported_schemes, params.integrator_type))
          return new LowStorageExplicitRungeKuttaIntegrator<number, PDEOperator>(params, timer);
      }
    if (Utils::contains(one_step_theta_supported_schemes, params.integrator_type))
      return new OneStepTheta<number, PDEOperator>(params, linear_solver_data);
    DEAL_II_NOT_IMPLEMENTED();
  }

} // namespace MeltPoolDG