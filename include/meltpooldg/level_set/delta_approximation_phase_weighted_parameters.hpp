#pragma once
#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  BETTER_ENUM(
    DiracDeltaFunctionApproximationType,
    char,
    norm_of_indicator_gradient, // use δ = ||∇ϕ|| as approximation for the Dirac delta function
    // with the heaviside representation of the level set ϕ
    phase_weighted_delta,        // see DeltaApproximationPhaseWeighted
    quad_phase_weighted_delta,   // see DeltaApproximationQuadPhaseWeighted
    double_phase_weighted_delta, // see DeltaApproximationDoublePhaseWeighted
    delta_weighted_consistent_with_evaporation
    // see DeltaApproximationPhaseWeightedConsistentWithEvaporation
  )

  template <typename number = double>
  struct DeltaApproximationPhaseWeightedData
  {
    DiracDeltaFunctionApproximationType type =
      DiracDeltaFunctionApproximationType::norm_of_indicator_gradient;
    number gas_phase_weight     = 1.0;
    number heavy_phase_weight   = 1.0;
    number gas_phase_weight_2   = 1.0;
    number heavy_phase_weight_2 = 1.0;

    void
    add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("dirac delta function approximation");
      {
        prm.add_parameter("type", type, "Choose how to smear a parameter over the interface.");
        prm.add_parameter(
          "gas phase weight",
          gas_phase_weight,
          "If >>> dirac delta function approximation type <<< is set to >>> phase_weighted_delta <<< "
          "this parameter controls the weight of the gas phase (level set = -1).");
        prm.add_parameter(
          "heavy phase weight",
          heavy_phase_weight,
          "If >>> dirac delta function approximation type <<< is set to >>> phase_weighted_delta <<< "
          "this parameter controls the weight of the heavy liquid/solid phase (level set = 1).");
        prm.add_parameter(
          "gas phase weight 2",
          gas_phase_weight_2,
          "If >>> dirac delta function approximation type <<< is set to >>> double_phase_weighted_delta <<< "
          "this parameter controls the second weight of the gas phase (level set = -1).");
        prm.add_parameter(
          "heavy phase weight 2",
          heavy_phase_weight_2,
          "If >>> dirac delta function approximation type <<< is set to >>> double_phase_weighted_delta <<< "
          "this parameter controls the second weight of the heavy liquid/solid phase (level set = 1).");
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
