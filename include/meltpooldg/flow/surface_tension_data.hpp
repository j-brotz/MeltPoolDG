#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number = double>
  struct SurfaceTensionData
  {
    number surface_tension_coefficient                       = 0.0;
    number temperature_dependent_surface_tension_coefficient = 0.0;
    number reference_temperature                             = dealii::numbers::invalid_double;
    number coefficient_residual_fraction                     = 0.0;
    bool   zero_surface_tension_in_solid                     = false;

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