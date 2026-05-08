#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_model_pressure_aware.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace MeltPoolDG;

int
main()
{
  using number = double;

  // Material parameters for Ti-6Al-4V at 1 bar atmospheric pressure
  typename Evaporation::EvaporationData<number>::PressureAwareData pressure_aware_data;
  pressure_aware_data.Km                   = {3.99601e-1, -4.11747e-7, 3.61321e-7, 4.19493e-10};
  pressure_aware_data.ambient_gas_pressure = 1.e5;

  constexpr number boiling_temperature     = 3315;
  constexpr number latent_heat_evaporation = 8.84e6;

  Evaporation::EvaporationModelPressureAware<number> evap_model_pressure_aware(
    pressure_aware_data, boiling_temperature, latent_heat_evaporation);

  std::vector<number> T_vec = {3100., 3300., 3500.};

  // Print header
  std::cout << std::left << std::setw(10) << "T" << std::setw(10) << "m_dot" << std::setw(10)
            << "dm_dot" << std::endl;

  for (number T : T_vec)
    {
      const number m_dot = evap_model_pressure_aware.local_compute_evaporative_mass_flux(T);
      const number dm_dot =
        evap_model_pressure_aware.local_compute_evaporative_mass_flux_derivative(T);

      // Print values
      std::cout << std::left << std::setw(10) << T << std::setw(12) << m_dot << std::setw(10)
                << dm_dot << std::endl;
    }
}
