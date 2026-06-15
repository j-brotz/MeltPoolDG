#include <deal.II/base/vectorization.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/phase_change/evaporative_cooling.templates.hpp>

#include <iostream>

using namespace MeltPoolDG::Evaporation;

int
main()
{
  using number = double;
  using namespace MeltPoolDG;

  EvaporationData<number> evapor_data;
  MaterialData<number>    material_data;

  material_data.boiling_temperature           = 3133.0;
  material_data.molar_mass                    = 0.0478;
  material_data.latent_heat_of_evaporation    = 8.84e6;
  material_data.liquid.specific_heat_capacity = 1000.0;

  evapor_data.evaporative_mass_flux_model = EvaporationModelType::saturated_vapor_pressure;

  evapor_data.recoil.enable                 = true;
  evapor_data.recoil.ambient_gas_pressure   = 1e5;
  evapor_data.recoil.activation_temperature = 1900.0;

  evapor_data.evaporative_cooling.enable_linear_activation_ramp               = true;
  evapor_data.evaporative_cooling.activation_temperature                      = 1900.0;
  evapor_data.evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux = "false";

  evapor_data.recoil.post(material_data);

  const number T_ac = evapor_data.evaporative_cooling.activation_temperature;
  const number T_v  = material_data.boiling_temperature;

  {
    EvaporativeCooling<number> evaporative_cooling(evapor_data, material_data, true);

    std::cout << "With activation ramp" << std::endl;

    for (const auto T : {T_ac - 10.0, T_ac, 0.5 * (T_ac + T_v), T_v, T_v + 10.0})
      std::cout << "T = " << T
                << ", evaporative cooling = " << evaporative_cooling.compute_evaporative_cooling(T)
                << ", derivative = "
                << evaporative_cooling
                     .compute_evaporative_cooling_derivative_with_temperature_dependent_mass_flux(T)
                << std::endl;
  }

  {
    evapor_data.evaporative_cooling.enable_linear_activation_ramp = false;

    EvaporativeCooling<number> evaporative_cooling_no_ramp(evapor_data, material_data, true);

    std::cout << "\nWithout activation ramp" << std::endl;

    for (const auto T : {T_ac - 10.0, T_ac, 0.5 * (T_ac + T_v), T_v, T_v + 10.0})
      std::cout << "T = " << T << ", evaporative cooling = "
                << evaporative_cooling_no_ramp.compute_evaporative_cooling(T) << ", derivative = "
                << evaporative_cooling_no_ramp
                     .compute_evaporative_cooling_derivative_with_temperature_dependent_mass_flux(T)
                << std::endl;
  }
  return 0;
}
