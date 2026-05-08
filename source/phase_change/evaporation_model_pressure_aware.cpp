#include <deal.II/base/numbers.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/phase_change/evaporation_model_pressure_aware.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.templates.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

#include <cmath>

namespace MeltPoolDG::Evaporation
{
  template <typename number>
  EvaporationModelPressureAware<number>::EvaporationModelPressureAware(
    const typename EvaporationData<number>::PressureAwareData &pressure_aware_data,
    const number                                               boiling_temperature,
    const number                                               latent_heat_evaporation)
    : pressure_aware_data(pressure_aware_data)
    , boiling_temperature(boiling_temperature)
    , latent_heat_evaporation(latent_heat_evaporation)
    , Km(pressure_aware_data.Km)
    , ambient_gas_pressure(pressure_aware_data.ambient_gas_pressure)
  {}


  template <typename number>
  number
  EvaporationModelPressureAware<number>::local_compute_evaporative_mass_flux(const number T) const
  {
    if (T < 1e-12)
      return 0.0;

    else
      {
        number mass_flux = 0.0;
        for (int i = 0; i < static_cast<int>(Km.size()); ++i)
          {
            mass_flux += Km[i] * std::pow((T - boiling_temperature), i + 1);
          }
        return mass_flux;
      }
  }

  template <typename number>
  dealii::VectorizedArray<number>
  EvaporationModelPressureAware<number>::local_compute_evaporative_mass_flux_vec(
    const dealii::VectorizedArray<number> &T) const
  {
    const dealii::VectorizedArray<number> dT        = T - boiling_temperature;
    dealii::VectorizedArray<number>       mass_flux = dealii::make_vectorized_array<number>(0.0);
    for (int i = 0; i < static_cast<int>(Km.size()); ++i)
      {
        mass_flux +=
          dealii::make_vectorized_array<number>(Km[i]) * std::pow(dT, static_cast<number>(i + 1));
      }
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(T,
                                                                             1e-12,
                                                                             0.0,
                                                                             mass_flux);
  }

  template <typename number>
  number
  EvaporationModelPressureAware<number>::local_compute_evaporative_mass_flux_derivative(
    const number T) const
  {
    if (T < 1e-12)
      return 0.0;

    else
      {
        number d_mass_flux = 0.0;
        for (int i = 0; i < static_cast<int>(Km.size()); ++i)
          {
            d_mass_flux += Km[i] * (i + 1) * std::pow((T - boiling_temperature), i);
          }
        return d_mass_flux;
      }
  }

  template <typename number>
  dealii::VectorizedArray<number>
  EvaporationModelPressureAware<number>::local_compute_evaporative_mass_flux_vec_derivative(
    const dealii::VectorizedArray<number> &T) const
  {
    const dealii::VectorizedArray<number> dT =
      T - dealii::make_vectorized_array<number>(boiling_temperature);
    dealii::VectorizedArray<number> mass_flux = dealii::make_vectorized_array<number>(0.0);
    for (int i = 0; i < static_cast<int>(Km.size()); ++i)
      {
        mass_flux += dealii::make_vectorized_array<number>(Km[i] * (i + 1)) *
                     std::pow(dT, static_cast<number>(i));
      }
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(T,
                                                                             1e-12,
                                                                             0.0,
                                                                             mass_flux);
  }

  template class EvaporationModelPressureAware<double>;
} // namespace MeltPoolDG::Evaporation
