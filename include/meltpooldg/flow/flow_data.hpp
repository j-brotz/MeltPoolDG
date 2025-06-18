#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/flow/darcy_damping_data.hpp>
#include <meltpooldg/flow/surface_tension_data.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Collection of general flow parameters.
   */
  template <typename number>
  struct FlowData
  {
    /// Gravity constant in m/s²
    number gravity = 0.0;

    /// Surface tension data
    SurfaceTensionData<number> surface_tension;

    /// Darcy damping data
    DarcyDampingData<number> darcy_damping;

    /**
     * @brief Add flow parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * @brief Post-process flow parameters.
     *
     * @param material Material data struct.
     */
    void
    post(const MaterialData<number> &material);

    /**
     * @brief Validates flow parameters.
     *
     * @param curv_enable Flag indicating whether curvature computation is enabled.
     */
    void
    check_input_parameters(const bool curv_enable) const;
  };
} // namespace MeltPoolDG::Flow
