#pragma once
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_model_base.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   * @brief This class implements the evaporative mass flux computed with
   * pressure-aware boundary conditions, as presented in
   * Refined Formulations of Resolved Vapor Flow and Unresolved
   * Recoil Pressure Models for Rapid Evaporation in Metal Additive
   * Manufacturing under Elevated Pressure
   *
   */
  template <typename number>
  class EvaporationModelPressureAware : public EvaporationModelBase<number>
  {
  private:
    /**
     * @param Km  Fitting parameters for empirical correlations in the free-surface model
     * @param ambient_gas_pressure Ambient gas pressure (build chamber).
     * @param latent_heat_of_evaporation Latent heat of evaporation.
     * @param boiling_temperature Boiling temperature at given build chamber pressure level.
     */
    const typename EvaporationData<number>::PressureAwareData pressure_aware_data;
    const number                                              boiling_temperature;
    const number                                              latent_heat_evaporation;

    std::vector<number> Km;
    const number        ambient_gas_pressure;

  public:
    EvaporationModelPressureAware(
      const typename EvaporationData<number>::PressureAwareData &pressure_aware_data,
      const number                                               boiling_temperature,
      const number                                               latent_heat_evaporation);

    /**
     * @brief The evaporative mass flux is computed as
     *
     * .       Nm-1
     * m(T) =   ∑    Kₘ,ᵢ · (T - Tᵥ(pᵍ))ⁱ⁺¹
     *         i=0
     *
     * @param T Melt surface temperature.
     */
    number
    local_compute_evaporative_mass_flux(const number T) const final;

    dealii::VectorizedArray<number>
    local_compute_evaporative_mass_flux_vec(const dealii::VectorizedArray<number> &T) const final;

    /**
     * @brief Compute the derivative of the evaporative mass flux as
     *
     *    .
     *  d m     Nm-1
     * ----- =   ∑    Kₘ,ᵢ · (i+1) · (T - Tᵥ(pᵍ))ⁱ
     *  d T     i=0
     *
     * @param T Melt surface temperature.
     */
    number
    local_compute_evaporative_mass_flux_derivative(const number T) const final;

    dealii::VectorizedArray<number>
    local_compute_evaporative_mass_flux_vec_derivative(
      const dealii::VectorizedArray<number> &T) const final;
  };
} // namespace MeltPoolDG::Evaporation
