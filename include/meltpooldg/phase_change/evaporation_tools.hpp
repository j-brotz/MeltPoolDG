#pragma once

#include <deal.II/base/vectorization.h>

#include <cmath>

namespace MeltPoolDG::Evaporation
{
  /**
   * @brief Calculate the saturated vapor pressure at a certain vapor temperature according to the
   * Clausius-Clapeyron equation.
   *
   * @param T_sat Saturated vapor temperature.
   * @param boiling_temperature Boiling temperature at atmospheric pressure conditions.
   * @param ambient_gas_pressure Ambient gas pressure.
   * @param temperature_constant Temperature constant; It should be
   * computed from latent_heat_of_evaporation * molar_mass / universal_gas_constant.
   *
   * @return Saturated vapor pressure at the provided saturated vapor temperature @p T_sat.
   *
   * @note The template parameter @tparam number_2 allows a vectorized computation.
   */
  template <typename number, typename number_2>
  inline number_2
  compute_saturated_gas_pressure(const number_2 &T_sat,
                                 const number    boiling_temperature,
                                 const number    ambient_gas_pressure,
                                 const number    temperature_constant)
  {
    const number T_v = boiling_temperature;
    return ambient_gas_pressure * std::exp(-temperature_constant * (1. / T_sat - 1. / T_v));
  }

} // namespace MeltPoolDG::Evaporation
