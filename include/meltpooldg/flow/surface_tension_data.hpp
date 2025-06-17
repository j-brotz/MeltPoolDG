#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>


namespace MeltPoolDG::Flow
{
  /**
   * @enum RegularizedSurfaceTensionTemperatureEvaluationType
   *
   * @brief Enumeration of temperature evaluation strategies for regularized surface tension.
   */
  BETTER_ENUM(RegularizedSurfaceTensionTemperatureEvaluationType,
              char,
              // The flux distributed in the interfacial zone is computed based on local values
              // evaluated at the quadrature points.
              local_value,
              // The flux distributed in the interfacial zone is computed based on values evaluated
              // at the projected quadrature points to the level set = 0 isosurface.
              interface_value)

  /**
   * @brief Collection of parameters related to surface tension effects.
   */
  template <typename number>
  struct SurfaceTensionData
  {
    /// Surface tension coefficient
    number surface_tension_coefficient = 0.0;

    /// Temperature dependent surface tension coefficient
    number temperature_dependent_surface_tension_coefficient = 0.0;

    /// Reference temperature
    number reference_temperature = numbers::invalid_double;

    /// Evaluation method for the temperature in the interfacial zone
    RegularizedSurfaceTensionTemperatureEvaluationType interface_temperature_evaluation_type =
      RegularizedSurfaceTensionTemperatureEvaluationType::local_value;

    /// Residual fraction parameter for the surface coefficient.
    number coefficient_residual_fraction = 0.0;

    /// Indicator whether surface tension should be omitted in the solid phase
    bool zero_surface_tension_in_solid = false;

    /// Data object for the phase-weighted delta approximation
    LevelSet::DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;

    /// Data struct related to the time step limit
    TimeIntegration::TimeStepLimitData<number> time_step_limit;

    /**
     * @brief Add surface tension related material parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * @brief Post-process surface tension parameters.
     *
     * @param material Material-specific parameters.
     */
    void
    post(const MaterialData<number> &material);

    /**
     * @brief Validates surface tension parameters.
     *
     * @param curv_enable Flag indicating whether curvature computation is enabled.
     */
    void
    check_input_parameters(const bool curv_enable) const;
  };
} // namespace MeltPoolDG::Flow
