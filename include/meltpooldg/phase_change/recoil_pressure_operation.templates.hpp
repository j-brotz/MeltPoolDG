#pragma once

#include <meltpooldg/phase_change/recoil_pressure_operation.hpp>
//
#include <deal.II/base/utilities.h>

#include <meltpooldg/phase_change/evaporation_tools.hpp>

#include <cmath>


namespace MeltPoolDG::Evaporation
{

  template <typename number>
  inline number
  RecoilPressureHybridModel<number>::compute_recoil_pressure_coefficient(
    const number T,
    const number m_dot,
    const number delta_coefficient) const
  {
    return recoil_phenomenological.compute_recoil_pressure_coefficient(T) -
           dealii::Utilities::fixed_power<2>(m_dot) * density_coeff * delta_coefficient;
  }

  template <typename number>
  inline dealii::VectorizedArray<number>
  RecoilPressureHybridModel<number>::compute_recoil_pressure_coefficient(
    const dealii::VectorizedArray<number> &T,
    const dealii::VectorizedArray<number> &m_dot,
    const dealii::VectorizedArray<number> &delta_coefficient) const
  {
    return recoil_phenomenological.compute_recoil_pressure_coefficient(T) -
           dealii::Utilities::fixed_power<2>(m_dot) * density_coeff * delta_coefficient;
  }

  template <typename number>
  inline dealii::VectorizedArray<number>
  RecoilPressurePhenomenologicalModel<number>::compute_recoil_pressure_coefficient(
    const dealii::VectorizedArray<number> &T) const
  {
    const number T_ac = recoil_data.activation_temperature;
    const number T_v  = boiling_temperature;

    dealii::VectorizedArray<number> full_recoil_pressure =
      recoil_data.pressure_coefficient *
      compute_saturated_gas_pressure(T,
                                     T_v,
                                     recoil_data.ambient_gas_pressure,
                                     recoil_data.temperature_constant);

    if (recoil_data.subtract_ambient_pressure)
      full_recoil_pressure -= recoil_data.ambient_gas_pressure;

    dealii::VectorizedArray<number> ramped_recoil_pressure =
      activation_ramp_derivative * (T - T_ac);

    dealii::VectorizedArray<number> recoil_pressure =
      compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(
        T,
        recoil_data.enable_linear_activation_ramp ? T_v : T_ac,
        full_recoil_pressure,
        recoil_data.enable_linear_activation_ramp ? ramped_recoil_pressure : 0.0);

    // Recoil pressure is constrained to remain non-negative.
    return compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(recoil_pressure,
                                                                                 0.0,
                                                                                 recoil_pressure,
                                                                                 0.0);
  }

  template <typename number>
  inline number
  RecoilPressurePhenomenologicalModel<number>::compute_recoil_pressure_coefficient(
    const number T) const
  {
    const number T_ac = recoil_data.activation_temperature;
    const number T_v  = boiling_temperature;

    // Recoil pressure is inactive below the activation temperature.
    if (T < T_ac)
      return 0.0;

    number recoil_pressure = 0;

    if (T >= T_v or not recoil_data.enable_linear_activation_ramp)
      {
        // Above the boiling temperature, or when the activation ramp is disabled,
        // compute the recoil pressure directly from the saturated gas pressure.
        recoil_pressure = recoil_data.pressure_coefficient *
                          compute_saturated_gas_pressure(T,
                                                         T_v,
                                                         recoil_data.ambient_gas_pressure,
                                                         recoil_data.temperature_constant);

        // Optionally use the pressure relative to the ambient gas pressure.
        if (recoil_data.subtract_ambient_pressure)
          recoil_pressure -= recoil_data.ambient_gas_pressure;
      }
    else
      {
        // Between the activation and boiling temperatures, linearly ramp the
        // recoil pressure from zero to its value at the boiling temperature.
        recoil_pressure = activation_ramp_derivative * (T - T_ac);
      }

    return std::max(recoil_pressure, 0.0);
  }



  template <typename number>
  inline number
  RecoilPressureModelPressureAware<number>::compute_recoil_pressure_coefficient(
    const number T) const
  {
    if (T < boiling_temperature)
      return 0.0;

    else
      {
        number       recoil_pressure = ambient_gas_pressure;
        const number T_diff          = T - boiling_temperature;

        for (size_t i = 0; i < Kp.size(); ++i)
          {
            recoil_pressure += Kp[i] * std::pow(T_diff, i + 2);
          }
        return recoil_pressure;
      }
  }

  template <typename number>
  inline dealii::VectorizedArray<number>
  RecoilPressureModelPressureAware<number>::compute_recoil_pressure_coefficient(
    const dealii::VectorizedArray<number> &T) const
  {
    const dealii::VectorizedArray<number> T_diff =
      T - dealii::make_vectorized_array<number>(boiling_temperature);
    dealii::VectorizedArray<number> recoil_pressure =
      dealii::make_vectorized_array<number>(ambient_gas_pressure);

    for (number i = 0; i < Kp.size(); ++i)
      {
        recoil_pressure += dealii::make_vectorized_array<number>(Kp[i]) * std::pow(T_diff, i + 2);
      }
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(T,
                                                                             boiling_temperature,
                                                                             0.0,
                                                                             recoil_pressure);
  }

} // namespace MeltPoolDG::Evaporation
