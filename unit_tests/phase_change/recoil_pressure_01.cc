#include <deal.II/base/vectorization.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.hpp>

#include <iostream>

using namespace MeltPoolDG::Evaporation;

int
main()
{
  using number = double;
  using namespace MeltPoolDG;

  RecoilPressureData<number> recoil_data;
  MaterialData<number>       material_data;
  material_data.boiling_temperature        = 3133;
  material_data.molar_mass                 = 0.0478;
  material_data.latent_heat_of_evaporation = 8.84e6;

  recoil_data.enable                        = true;
  recoil_data.ambient_gas_pressure          = 1e5;
  recoil_data.activation_temperature        = 1900.0;
  recoil_data.enable_linear_activation_ramp = true;
  recoil_data.subtract_ambient_pressure     = false;

  recoil_data.post(material_data);

  const number T_ac = recoil_data.activation_temperature;
  const number T_v  = material_data.boiling_temperature;
  const number h_v  = material_data.latent_heat_of_evaporation;
  const number M    = material_data.molar_mass;

  {
    RecoilPressurePhenomenologicalModel<number> model(recoil_data, T_v, M, h_v);

    std::cout << "With activation ramp" << std::endl;

    for (const auto T : {T_ac - 10.0, T_ac, 0.5 * (T_ac + T_v), T_v, T_v + 10.0})
      std::cout << "T = " << T
                << ", recoil pressure = " << model.compute_recoil_pressure_coefficient(T)
                << std::endl;
  }

  {
    recoil_data.enable_linear_activation_ramp = false;

    RecoilPressurePhenomenologicalModel<number> no_ramp_model(recoil_data, T_v, M, h_v);

    std::cout << "\nWithout activation ramp" << std::endl;

    for (const auto T : {T_ac - 10.0, T_ac, 0.5 * (T_ac + T_v), T_v, T_v + 10.0})
      std::cout << "T = " << T
                << ", recoil pressure = " << no_ramp_model.compute_recoil_pressure_coefficient(T)
                << std::endl;
  }

  {
    recoil_data.enable_linear_activation_ramp = false;
    recoil_data.subtract_ambient_pressure     = true;

    RecoilPressurePhenomenologicalModel<number> no_ramp_model(recoil_data, T_v, M, h_v);

    std::cout << "\nWithout activation ramp and with ambient pressure subtraction" << std::endl;

    for (const auto T :
         {T_ac - 10.0, T_ac, 0.5 * (T_ac + T_v), T_v, T_v + 100, T_v + 200, T_v + 300})
      std::cout << "T = " << T
                << ", recoil pressure = " << no_ramp_model.compute_recoil_pressure_coefficient(T)
                << std::endl;
  }

  return 0;
}
