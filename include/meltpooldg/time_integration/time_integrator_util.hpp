
#pragma once

#include <meltpooldg/time_integration/explicit_integrator_base.hpp>
#include <meltpooldg/time_integration/explicit_low_storage_runge_kutta_integrator.hpp>
#include <meltpooldg/time_integration/explicit_runge_kutta_integrator.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <memory>

namespace MeltPoolDG::TimeIntegration
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
  scheme_is_explicit(const TimeIntegratorSchemes scheme)
  {
    if (Utils::contains(StaticExplicitRungeKuttaIntegrator::supported_schemes, scheme))
      return true;
    if (Utils::contains(StaticExplicitLowStorageRungeKuttaIntegrator::supported_schemes, scheme))
      return true;
    return false;
  }

  /**
   * Factory function that creates and returns a unique pointer to an explicit time integrator
   * based on the scheme specified in @p TimeIntegratorData.
   *
   * @param params Contains the configuration details for the time integrator.
   * @param timer Timer passed to the constructor of the time integrator.
   *
   * @return A unique pointer to the appropriate explicit time integrator.
   * @throws An exception if the specified integration scheme is not supported.
   */
  template <typename Number, typename PDEOperator>
  std::unique_ptr<ExplicitIntegratorBase<Number, PDEOperator>>
  explicit_integrator_factory(const TimeIntegratorData &params, dealii::TimerOutput &timer)
  {
    if constexpr (ExplicitPDEOperator<PDEOperator,
                                      Number,
                                      dealii::LinearAlgebra::distributed::Vector<Number>>)
      {
        if (Utils::contains(LowStorageRungeKuttaIntegrator<Number, PDEOperator>::supported_schemes,
                            params.integrator_type))
          return std::make_unique<LowStorageRungeKuttaIntegrator<Number, PDEOperator>>(params,
                                                                                       timer);
        if (Utils::contains(ExplicitRungeKuttaIntegrator<Number, PDEOperator>::supported_schemes,
                            params.integrator_type))
          {
            return std::make_unique<ExplicitRungeKuttaIntegrator<Number, PDEOperator>>(params,
                                                                                       timer);
          }
      }
    DEAL_II_NOT_IMPLEMENTED();
  }
} // namespace MeltPoolDG::TimeIntegration