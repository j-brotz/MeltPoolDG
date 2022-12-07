#pragma once
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  BETTER_ENUM(
    DiracDeltaFunctionApproximationType,
    char,
    norm_of_indicator_gradient, // use δ = ||∇ϕ|| as approximation for the Dirac delta function
    // with the heaviside representation of the level set ϕ
    heaviside_phase_weighted,      // see DeltaApproximationHeavisidePhaseWeighted
    quad_heaviside_phase_weighted, // see DeltaApproximationQuadHeavisidePhaseWeighted
    heaviside_times_heaviside_phase_weighted,
    // see DeltaApproximationHeavisideTimesHeavisidePhaseWeighted
    reciprocal_phase_weighted,                 // see DeltaApproximationReciprocalPhaseWeighted
    reciprocal_times_heaviside_phase_weighted, // see
                                               // DeltaApproximationReciprocalTimesHeavisidePhaseWeighted
    heavy_phase_only // see DeltaApproximationHeavyPhaseOnly
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
    bool   auto_weights         = false;

    void
    add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("dirac delta function approximation");
      {
        prm.add_parameter("type", type, "Choose how to smear a parameter over the interface.");
        prm.add_parameter("auto weights",
                          auto_weights,
                          "Choose if weights should be computed automatically.");
        prm.add_parameter(
          "gas phase weight",
          gas_phase_weight,
          "If >>> dirac delta function approximation type <<< is set to any phase weighted option"
          "this parameter controls the (first) weight of the gas phase (level set = -1).");
        prm.add_parameter(
          "heavy phase weight",
          heavy_phase_weight,
          "If >>> dirac delta function approximation type <<< is set to any phase weighted option"
          "this parameter controls the (first) weight of the heavy phase (level set = 1).");
        prm.add_parameter(
          "gas phase weight 2",
          gas_phase_weight_2,
          "If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< "
          "this parameter controls the second weight of the gas phase (level set = -1).");
        prm.add_parameter(
          "heavy phase weight 2",
          heavy_phase_weight_2,
          "If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< "
          "this parameter controls the second weight of the heavy liquid/solid phase (level set = 1).");
      }
      prm.leave_subsection();
    }

    void
    set_parameters(const MaterialData<number> &mat)
    {
      if (auto_weights)
        {
          gas_phase_weight     = mat.first.density;
          heavy_phase_weight   = mat.second.density;
          gas_phase_weight_2   = mat.first.capacity;
          heavy_phase_weight_2 = mat.second.capacity;
        }
    }
  };
} // namespace MeltPoolDG
