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

  /**
   * Compute a linear activation scaling coefficient.
   *
   * The scaling coefficient is used to smoothly activate a quantity (e.g.,
   * recoil pressure or evaporative cooling) between an activation temperature
   * and a boiling temperature. The coefficient is defined as
   *
   * \f[
   * s(T) =
   * \begin{cases}
   * 0, & T \le T_{\mathrm{ac}}, \\
   * \dfrac{T - T_{\mathrm{ac}}}{T_{\mathrm{v}} - T_{\mathrm{ac}}},
   *    & T_{\mathrm{ac}} < T < T_{\mathrm{v}}, \\
   * 1, & T \ge T_{\mathrm{v}}.
   * \end{cases}
   * \f]
   *
   * where \f$T_{\mathrm{ac}}\f$ is the activation temperature and
   * \f$T_{\mathrm{v}}\f$ is the boiling temperature.
   *
   * @param T Temperature at which the scaling coefficient is evaluated.
   * @param T_ac Activation temperature.
   * @param T_v Boiling temperature.
   *
   * @return The scaling coefficient in the range \f$[0,1]\f$.
   */
  template <typename number>
  inline dealii::VectorizedArray<number>
  compute_linear_scaling_coeff(const dealii::VectorizedArray<number> &T,
                               const number                           T_ac,
                               const number                           T_v)
  {
    return compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      T,
      T_ac,
      0.0,
      compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(
        T, T_v, 1., (T - T_ac) / (T_v - T_ac)));
  }

  /**
   * Same as above, just for the vectorized case.
   */
  template <typename number>
  inline number
  compute_linear_scaling_coeff(const number T, const number T_ac, const number T_v)
  {
    return (T < T_ac) ? 0.0 : (T >= T_v) ? 1. : (T - T_ac) / (T_v - T_ac);
  }
} // namespace MeltPoolDG::Evaporation
