#include <meltpooldg/evaporation/recoil_pressure_data.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <typename number>
  void
  RecoilPressureData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("recoil pressure");
    {
      prm.add_parameter("ambient gas pressure",
                        ambient_gas_pressure,
                        "Ambient gas pressure for the recoil pressure model.");
      prm.add_parameter("pressure coefficient",
                        pressure_coefficient,
                        "Pressure coefficient for the recoil pressure model.",
                        Patterns::Double(0.0, 1.0));
      prm.add_parameter("temperature constant",
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


  template <typename number>
  void
  RecoilPressureData<number>::post(const MaterialData<number> &material)
  {
    // recoil pressure: set default value of activation temperature equal to the boiling
    // temperature
    if (dealii::numbers::is_invalid(activation_temperature))
      activation_temperature = material.boiling_temperature;

    // set automatic weights of asymmetric delta functions, if requested
    this->delta_approximation_phase_weighted.set_parameters(
      material, LevelSet::ParameterScaledInterpolationType::density);
  }
  template struct RecoilPressureData<double>;
} // namespace MeltPoolDG::Evaporation
