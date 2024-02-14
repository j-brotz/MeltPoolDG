#pragma once
#include <meltpooldg/level_set/delta_approximation_phase_weighted_parameters.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  // @todo: move to own file
  BETTER_ENUM(RegularizedRecoilPressureTemperatureEvaluationType,
              char,
              // The flux distributed in the interfacial zone is computed based on local values
              // evaluated at the quadrature points.
              local_value,
              // The flux distributed in the interfacial zone is computed based on values evaluated
              // at the projected quadrature points to the level set = 0 isosurface.
              interface_value)

  BETTER_ENUM(RecoilPressureModelType,
              char,
              // default: compute phenomenological recoil pressure
              //    p_v(T) = p_recoil_phenomenological(T)
              phenomenological,
              // hybrid recoil pressure model, considering the evaporation-induced velocity jump;
              // The pressure jump is computed from
              //
              //    p_v(T) = p_recoil_phenomenological(T) - mDot^2*(1/rho_g-1/rho_l).
              //
              // This ensures that the evaporation-induced pressure jump is the same as
              // in the phenomenological recoil pressure model.
              hybrid)

  template <typename number = double>
  struct RecoilPressureData
  {
    // recoil pressure constant
    // recommended as c_p = 0.56 * p_a with p_a being the ambient pressure (e.g. 100 Pa for air)
    number pressure_constant = 0.0;
    // temperature constant
    // recommended as c_T = h_v/R with the molar latent heat of evaporation h_v
    // and the universal gas constant R.
    number temperature_constant = 0.0;
    // activation temperature of the recoil pressure; must be smaller than or equal to the boiling
    // temperature; this parameter enables a smooth activation of the recoil pressure
    number activation_temperature =
      dealii::numbers::invalid_double; //@todo: introduce invalid_number
    // Choose how the recoil pressure flux across the interface should be computed:
    // * continuous: use the local temperature value
    // * interface_value: use the value at the interface (level set=0)
    RegularizedRecoilPressureTemperatureEvaluationType interface_distributed_flux_type =
      RegularizedRecoilPressureTemperatureEvaluationType::local_value;
    // Choose the delta-function for computing the continuum interface force.
    DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;
    // Choose the model type to compute the recoil pressure:
    //   * phenomenological (default)
    //   * consistent
    RecoilPressureModelType model_type = RecoilPressureModelType::phenomenological;


    void
    add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("recoil pressure");
      {
        prm.add_parameter("recoil pressure constant",
                          pressure_constant,
                          "Pressure constant for the recoil pressure model.");
        prm.add_parameter("recoil temperature constant",
                          temperature_constant,
                          "Temperature constant for the recoil pressure model.");
        prm.add_parameter("interface distributed flux type",
                          interface_distributed_flux_type,
                          "Type that determines how the recoil pressure force is computed in the "
                          "interfacial zone.");
        prm.add_parameter(
          "activation temperature",
          activation_temperature,
          "Activation temperature for the recoil pressure. It must be smaller than or equal to the "
          "boiling temperature. As default value, the boiling temperature is chosen.");
        delta_approximation_phase_weighted.add_parameters(prm);
        prm.add_parameter(
          "model type",
          model_type,
          "Choose the model to compute the recoil pressure coefficient: phenomenological "
          "or hybrid, in case there is also an evaporation-induced velocity jump.");
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG::Evaporation
