#include <meltpooldg/phase_change/recoil_pressure_data.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.templates.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace MeltPoolDG;

int
main()
{
  using number = double;

  // Material parameters for Ti-6Al-4V at 1 bar atmospheric pressure
  typename Evaporation::RecoilPressureData<number>::PressureAwareData pressure_aware_data;
  pressure_aware_data.Kp                   = {9.33135e-1, 2.26125e-4, 4.85206e-7};
  pressure_aware_data.ambient_gas_pressure = 1.e5;

  constexpr number molar_mass              = 4.26e-2;
  constexpr number boiling_temperature     = 3315;
  constexpr number latent_heat_evaporation = 9.70e6;

  Evaporation::RecoilPressureModelPressureAware<number> recoil_pressure_model_pressure_aware(
    pressure_aware_data, boiling_temperature, molar_mass, latent_heat_evaporation);

  std::vector<number> T_vec = {3100, 3500, 3800};

  // Print header
  std::cout << std::left << std::setw(10) << "T" << std::setw(10) << std::setprecision(5)
            << "p_recoil" << std::endl;

  for (number T : T_vec)
    {
      const number p_recoil =
        recoil_pressure_model_pressure_aware.compute_recoil_pressure_coefficient(T);

      // Print values
      std::cout << std::left << std::setw(10) << T << std::setw(10) << std::setprecision(5)
                << p_recoil << std::endl;
    }
}
