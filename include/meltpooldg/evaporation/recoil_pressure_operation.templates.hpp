#pragma once
#include <deal.II/base/vectorization.h>

#include <meltpooldg/evaporation/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

#include <cmath>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;
  /**
   * Compute saturated gas pressure.
   *
   * @param[in] T Temperature.
   * @param[in] boiling_temperature Boiling point.
   * @param[in] ambient_gas_pressure Ambient gas pressure.
   * @param[in] temperature_constant Temperature constant; It should be
   * computed from latent_heat_of_evaporation * molar_mass / universal_gas_constant.
   */
  template <typename number, typename number_2>
  inline number_2
  compute_saturated_gas_pressure(const number_2 &T,
                                 const number    boiling_temperature,
                                 const number    ambient_gas_pressure,
                                 const number    temperature_constant)
  {
    const number T_v = boiling_temperature;
    return ambient_gas_pressure * std::exp(-temperature_constant * (1. / T - 1. / T_v));
  }

  namespace internal
  {
    // In order to smoothly activate the recoil pressure, a scaling coefficient is computed and
    // multiplied with the recoil pressure coefficient
    //
    //                  -
    //                 |     0         if T <= T_ac
    //                 |
    //                 |  T - T_ac
    // scaling_coeff = |  --------     if T_ac < T < T_v
    //                 |  T_v - T_ac
    //                 |
    //                 |     1         if T >= T_v
    //                  -
    //
    // with @p T_ac the activation temperature of the recoil pressure and @p T_v the boiling temperature.

    template <typename number>
    inline VectorizedArray<number>
    compute_scaling_coeff(const VectorizedArray<number> &T, const number T_ac, const number T_v)
    {
      return compare_and_apply_mask<SIMDComparison::less_than>(
        T,
        T_ac,
        0.0,
        compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
          T, T_v, 1., (T - T_ac) / (T_v - T_ac)));
    }

    template <typename number>
    inline number
    compute_scaling_coeff(const number T, const number T_ac, const number T_v)
    {
      return (T < T_ac) ? 0.0 : (T >= T_v) ? 1. : (T - T_ac) / (T_v - T_ac);
    }
  } // namespace internal

  template <typename number>
  inline number
  RecoilPressureHybridModel<number>::compute_recoil_pressure_coefficient(
    const number T,
    const number m_dot,
    const number delta_coefficient) const
  {
    return recoil_phenomenological.compute_recoil_pressure_coefficient(T) -
           Utilities::fixed_power<2>(m_dot) * density_coeff * delta_coefficient;
  }

  template <typename number>
  inline VectorizedArray<number>
  RecoilPressureHybridModel<number>::compute_recoil_pressure_coefficient(
    const VectorizedArray<number> &T,
    const VectorizedArray<number> &m_dot,
    const VectorizedArray<number> &delta_coefficient) const
  {
    return recoil_phenomenological.compute_recoil_pressure_coefficient(T) -
           Utilities::fixed_power<2>(m_dot) * density_coeff * delta_coefficient;
  }

  template <typename number>
  inline VectorizedArray<number>
  RecoilPressurePhenomenologicalModel<number>::compute_recoil_pressure_coefficient(
    const VectorizedArray<number> &T) const
  {
    const number T_ac = recoil_data.activation_temperature;
    const number T_v  = boiling_temperature;

    const VectorizedArray<number> scaling_coeff = internal::compute_scaling_coeff(T, T_ac, T_v);

    return scaling_coeff * recoil_data.pressure_coefficient *
           compute_saturated_gas_pressure(T,
                                          T_v,
                                          recoil_data.ambient_gas_pressure,
                                          recoil_data.temperature_constant);
  }

  template <typename number>
  inline number
  RecoilPressurePhenomenologicalModel<number>::compute_recoil_pressure_coefficient(
    const number T) const
  {
    const number T_ac = recoil_data.activation_temperature;
    const number T_v  = boiling_temperature;

    const number scaling_coeff = internal::compute_scaling_coeff(T, T_ac, T_v);

    return scaling_coeff * recoil_data.pressure_coefficient *
           compute_saturated_gas_pressure(T,
                                          T_v,
                                          recoil_data.ambient_gas_pressure,
                                          recoil_data.temperature_constant);
  }

} // namespace MeltPoolDG::Evaporation
