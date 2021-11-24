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
    phase_weighted_delta, // approximate the Dirac delta function with
                          // δ = ||∇ϕ|| ( w_g (1-ϕ) + w_h ϕ ) 2 / ( w_g + w_h )
                          // with the weight of the gas phase w_g and the weight of the heavy phase
                          // w_h
    // with the heaviside representation of the level set ϕ
    quad_phase_weighted_delta, // approximate the Dirac delta function with
                               // δ = ||∇ϕ|| ( w_g (1-ϕ) + w_h ϕ )² 3 / ( w_g² + w_g w_h + w_h² )
                               // with the weight of the gas phase w_g and the weight of the heavy
                               // phase w_h
    double_phase_weighted_delta // approximate the Dirac delta function with
                                //      6 ||∇ϕ|| ( w_1g (1-ϕ) + w_1h ϕ )( w_2g (1-ϕ) + w_2h ϕ )
                                // δ = ---------------------------------------------------------
                                //         2 w_1g w_2g + w_1g w_2h + w_1h w_2g + 2 w_1h w_2h
                                // with the weights of the gas phase w_1g and w_2g and the weights
                                // of the heavy phase w_1h and w_2h
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
  };

  template <typename number>
  void
  add_parameter_delta_approximation_phase_weighted_data(
    ParameterHandler &                           prm,
    DeltaApproximationPhaseWeightedData<number> &data)
  {
    prm.enter_subsection("dirac delta function approximation");
    {
      prm.add_parameter("type", data.type, "Choose how to smear a parameter over the interface.");
      prm.add_parameter(
        "gas phase weight",
        data.gas_phase_weight,
        "If >>> dirac delta function approximation type <<< is set to >>> phase_weighted_delta <<< "
        "this parameter controls the weight of the gas phase (level set = -1).");
      prm.add_parameter(
        "heavy phase weight",
        data.heavy_phase_weight,
        "If >>> dirac delta function approximation type <<< is set to >>> phase_weighted_delta <<< "
        "this parameter controls the weight of the heavy liquid/solid phase (level set = 1).");
      prm.add_parameter(
        "gas phase weight 2",
        data.gas_phase_weight_2,
        "If >>> dirac delta function approximation type <<< is set to >>> double_phase_weighted_delta <<< "
        "this parameter controls the second weight of the gas phase (level set = -1).");
      prm.add_parameter(
        "heavy phase weight 2",
        data.heavy_phase_weight_2,
        "If >>> dirac delta function approximation type <<< is set to >>> double_phase_weighted_delta <<< "
        "this parameter controls the second weight of the heavy liquid/solid phase (level set = 1).");
    }
    prm.leave_subsection();
  }
} // namespace MeltPoolDG
