#include <deal.II/base/numbers.h>

#include <meltpooldg/phase_change/evaporation_model_saturated_vapor_pressure.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.templates.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

#include <cmath>

namespace MeltPoolDG::Evaporation
{
  template <typename number>
  EvaporationModelSaturatedVaporPressure<number>::EvaporationModelSaturatedVaporPressure(
    const RecoilPressureData<number> &recoil_data,
    const number                      boiling_temperature,
    const number                      molar_mass,
    const number                      latent_heat_evaporation)
    : recoil_data(recoil_data)
    , boiling_temperature(boiling_temperature)
    , sticking_constant(recoil_data.sticking_constant)
    , molar_mass(molar_mass)
    , latent_heat_evaporation(latent_heat_evaporation)
    , Cm(molar_mass / (2. * dealii::numbers::PI * PhysicalConstants::universal_gas_constant))
  {}


  template <typename number>
  number
  EvaporationModelSaturatedVaporPressure<number>::local_compute_evaporative_mass_flux(
    const number T) const
  {
    if (std::abs(T) < 1e-12)
      return 0.0;

    return 0.82 * sticking_constant *
           compute_saturated_gas_pressure(T,
                                          boiling_temperature,
                                          recoil_data.ambient_gas_pressure,
                                          recoil_data.temperature_constant) *
           std::sqrt(Cm / T);
  }
  template <typename number>
  dealii::VectorizedArray<number>
  EvaporationModelSaturatedVaporPressure<number>::local_compute_evaporative_mass_flux_vec(
    const dealii::VectorizedArray<number> &T) const
  {
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      std::abs(T),
      1e-12,
      0.0,
      0.82 * sticking_constant *
        compute_saturated_gas_pressure(T,
                                       boiling_temperature,
                                       recoil_data.ambient_gas_pressure,
                                       recoil_data.temperature_constant) *
        std::sqrt(Cm / T));
  }

  template <typename number>
  number
  EvaporationModelSaturatedVaporPressure<number>::local_compute_evaporative_mass_flux_derivative(
    const number T) const
  {
    if (std::abs(T) < 1e-12)
      return 0.0;

    return local_compute_evaporative_mass_flux(T) *
           (recoil_data.temperature_constant / (T * T) - 0.5 / T);
  }

  template <typename number>
  dealii::VectorizedArray<number>
  EvaporationModelSaturatedVaporPressure<number>::
    local_compute_evaporative_mass_flux_vec_derivative(
      const dealii::VectorizedArray<number> &T) const
  {
    // TODO this is not the derivative if temperature is between activation und boiling temperature
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      std::abs(T),
      1e-12,
      0.0,
      local_compute_evaporative_mass_flux_vec(T) *
        (recoil_data.temperature_constant / (T * T) - 0.5 / T));
  }

  template class EvaporationModelSaturatedVaporPressure<double>;
} // namespace MeltPoolDG::Evaporation
