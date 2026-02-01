#pragma once

#include <meltpooldg/flow/compressible_flow_utils.hpp>

#include <memory>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Interface of the compressible flow operator interacting with the compressible flow
   * operation.
   */
  template <int dim, typename number>
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

    virtual void
    add_external_force(
      std::shared_ptr<ExternalFlowForce<dim, number>>         external_force_residuum,
      std::shared_ptr<ExternalFlowForceJacobian<dim, number>> external_force_jacobian) = 0;
  };
} // namespace MeltPoolDG::Flow
