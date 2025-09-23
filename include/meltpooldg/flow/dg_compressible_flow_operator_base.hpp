#pragma once


#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Interface of the compressible flow operator interacting with the compressible flow
   * operation.
   */
  template <typename number>
  class DGCompressibleFlowOperatorBase
  {
  public:
    virtual ~DGCompressibleFlowOperatorBase() = default;

    /**
     * @brief Advances solver by a single time step.
     *
     * This function performs a single time step of size @p time_step starting from the solution
     * at time @p time. For a detailed description refer to the documentation of the derived
     * classes.
     */
    virtual void
    advance_time_step(number time, number time_step) = 0;

    /**
     * @brief Reinit the operator.
     *
     * This function is intended to be called every time at which the data structure needs to be
     * updated.
     */
    virtual void
    reinit() = 0;
  };
} // namespace MeltPoolDG::Flow
