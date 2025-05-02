/**
 * @brief Interface of the compressible flow operator interacting with the compressible flow operation.
 */

#pragma once


#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  class DGCompressibleFlowOperatorBase
  {
  public:
    virtual ~DGCompressibleFlowOperatorBase() = default;

    /**
     * Reinit the operator. This function is intended to be called every time at which the data
     * structure needs to be updated.
     */
    virtual void
    reinit() = 0;

    /**
     * Creates and returns the time integrator object which can be used in combination with the own
     * operator type.
     *
     * @param time_integrator_data Reference to the time integrator data object.
     *
     * @return Unique pointer to a time integrator which is templated on the own operator type.
     *
     * @throws If the time integrator type in the time integrator data does not fit to the current
     * operator type, e.g. if the operator type is implicit but the required time integration scheme
     * is an explicit scheme.
     */
    virtual std::unique_ptr<TimeIntegration::TimeIntegratorBase<number>>
    make_application_specific_time_integrator(
      const TimeIntegration::TimeIntegratorData<number> &time_integrator_data) = 0;
  };
} // namespace MeltPoolDG::Flow
