#pragma once
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  BETTER_ENUM(
    DiracDeltaFunctionApproximationType,
    char,
    norm_of_indicator_gradient, // use δ = ||∇ϕ|| as approximation for the Dirac delta function
    // with the heaviside representation of the level set ϕ
    heaviside_phase_weighted, // see DeltaApproximationHeavisidePhaseWeighted
    heaviside_times_heaviside_phase_weighted,
    // see DeltaApproximationHeavisideTimesHeavisidePhaseWeighted
    reciprocal_phase_weighted,                 // see DeltaApproximationReciprocalPhaseWeighted
    reciprocal_times_heaviside_phase_weighted, // see
                                               // DeltaApproximationReciprocalTimesHeavisidePhaseWeighted
    heavy_phase_only // see DeltaApproximationHeavyPhaseOnly
  )

  BETTER_ENUM(ParameterScaledInterpolationType,
              char,
              volume_specific_heat_capacity,
              specific_heat_capacity_times_density,
              density)

  template <typename number = double>
  struct DeltaApproximationPhaseWeightedData
  {
    DiracDeltaFunctionApproximationType type =
      DiracDeltaFunctionApproximationType::norm_of_indicator_gradient;
    number gas_phase_weight     = 1.0;
    number heavy_phase_weight   = 1.0;
    number gas_phase_weight_2   = 1.0;
    number heavy_phase_weight_2 = 1.0;
    bool   auto_weights         = false;

    void
    add_parameters(ParameterHandler &prm);

    void
    set_parameters(const MaterialData<number>             &material,
                   const ParameterScaledInterpolationType &interpolation_type);
  };
} // namespace MeltPoolDG::LevelSet
