#include <meltpooldg/phase_change/evaporation_model_knight.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace MeltPoolDG;

int
main()
{
  using number = double;

  constexpr number atmospheric_pressure = 1.e5;

  // Material parameters for Ti-6Al-4V
  constexpr number boiling_temperature_at_atmospheric_pressure = 3133;
  constexpr number latent_heat_of_evaporation                  = 8.84e6;
  constexpr number specific_gas_constant                       = 173.93;
  constexpr number specific_heat_ratio_vapor                   = 5. / 3.;

  Evaporation::EvaporationModelKnight<number> evap_model_knight(
    atmospheric_pressure,
    boiling_temperature_at_atmospheric_pressure,
    latent_heat_of_evaporation,
    specific_gas_constant,
    specific_heat_ratio_vapor);

  std::vector<number> Ma_gas_vec   = {0.1, 0.5, 1.};
  std::vector<number> T_liquid_vec = {3100., 3300., 3500.};

  // Print header
  std::cout << std::left << std::setw(10) << "Ma" << std::setw(12) << "T_liquid" << std::setw(10)
            << "m_dot" << std::setw(10) << "T_jump" << std::endl;

  // run test
  for (number Ma_gas : Ma_gas_vec)
    {
      for (number T_liquid : T_liquid_vec)
        {
          evap_model_knight.reinit(T_liquid, Ma_gas);
          const number m_dot  = evap_model_knight.get_evaporative_mass_flux();
          const number T_jump = evap_model_knight.get_temperature_jump();

          // Print values
          std::cout << std::left << std::setw(10) << Ma_gas << std::setw(12) << T_liquid
                    << std::setw(10) << m_dot << std::setw(10) << T_jump << std::endl;
        }
    }
}
