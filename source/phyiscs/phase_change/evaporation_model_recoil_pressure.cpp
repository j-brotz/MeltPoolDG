#include <meltpooldg/phase_change/evaporation_model_recoil_pressure.hpp>
//
#include <deal.II/base/numbers.h>

#include <meltpooldg/utilities/physical_constants.hpp>

#include <cmath>

namespace MeltPoolDG::Evaporation
{
  template <typename number>
  EvaporationModelRecoilPressure<number>::EvaporationModelRecoilPressure(
    const RecoilPressureData<number> &recoil_data,
    const number                      boiling_temperature,
    const number                      molar_mass,
    const number                      latent_heat_evaporation)
    : recoil_model(recoil_data, boiling_temperature, molar_mass, latent_heat_evaporation)
    , sticking_constant(recoil_data.sticking_constant)
    , Cm(molar_mass / (2. * dealii::numbers::PI * PhysicalConstants::universal_gas_constant))
    , temperature_constant(recoil_data.temperature_constant)
  {}

  template <typename number>
  number
  EvaporationModelRecoilPressure<number>::local_compute_evaporative_mass_flux(const number T) const
  {
    if (T < 1e-12)
      return 0.0;
    return 0.82 * sticking_constant * recoil_model.compute_recoil_pressure_coefficient(T) *
           std::sqrt(Cm / T); //@todo: replace recoil pressure by saturated vapor pressure
  }

  template <typename number>
  dealii::VectorizedArray<number>
  EvaporationModelRecoilPressure<number>::local_compute_evaporative_mass_flux_vec(
    const dealii::VectorizedArray<number> &T) const
  {
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      T,
      1e-12,
      0.0,
      0.82 * sticking_constant * recoil_model.compute_recoil_pressure_coefficient(T) *
        std::sqrt(Cm / T));
  }

  template <typename number>
  number
  EvaporationModelRecoilPressure<number>::local_compute_evaporative_mass_flux_derivative(
    const number T) const
  {
    if (T < 1e-12)
      return 0.0;

    return local_compute_evaporative_mass_flux(T) * (temperature_constant / (T * T) - 0.5 / T);
  }

  template <typename number>
  dealii::VectorizedArray<number>
  EvaporationModelRecoilPressure<number>::local_compute_evaporative_mass_flux_vec_derivative(
    const dealii::VectorizedArray<number> &T) const
  {
    // TODO this is not the derivative if temperature is between activation und boiling temperature
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      T,
      1e-12,
      0.0,
      local_compute_evaporative_mass_flux_vec(T) * (temperature_constant / (T * T) - 0.5 / T));
  }

  template class EvaporationModelRecoilPressure<double>;
} // namespace MeltPoolDG::Evaporation
