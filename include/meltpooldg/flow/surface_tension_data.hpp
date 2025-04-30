#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>


namespace MeltPoolDG::Flow
{
  BETTER_ENUM(RegularizedSurfaceTensionTemperatureEvaluationType,
              char,
              // The flux distributed in the interfacial zone is computed based on local values
              // evaluated at the quadrature points.
              local_value,
              // The flux distributed in the interfacial zone is computed based on values evaluated
              // at the projected quadrature points to the level set = 0 isosurface.
              interface_value)


  template <typename number>
  struct SurfaceTensionData
  {
    number surface_tension_coefficient                       = 0.0;
    number temperature_dependent_surface_tension_coefficient = 0.0;
    number reference_temperature                             = dealii::numbers::invalid_double;
    RegularizedSurfaceTensionTemperatureEvaluationType interface_temperature_evaluation_type =
      RegularizedSurfaceTensionTemperatureEvaluationType::local_value;
    number coefficient_residual_fraction = 0.0;
    bool   zero_surface_tension_in_solid = false;

    LevelSet::DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;
    TimeStepLimitData<number>                             time_step_limit;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const MaterialData<number> &material);

    void
    check_input_parameters(const bool curv_enable) const;
  };
} // namespace MeltPoolDG::Flow
