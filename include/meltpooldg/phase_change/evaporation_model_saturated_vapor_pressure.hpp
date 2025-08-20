#pragma once
#include <meltpooldg/phase_change/evaporation_model_base.hpp>
#include <meltpooldg/phase_change/recoil_pressure_data.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   * This class implements the evaporative mass flux computed from the
   * saturated vapor pressure as presented in, e.g.,
   *
   * S.A. Khairallah, A.T. Anderson, A. Rubenchik, W.E. King,
   * Laser powder-bed fusion additive manufacturing: Physics of complex
   * melt flow and formation mechanisms of pores, spatter, and denudation
   * zones, Acta Mater. 108 (2016) 36–45.
   * DOI: 10.1016/j.actamat.2016.02.014.
   *
   */
  template <typename number>
  class EvaporationModelSaturatedVaporPressure : public EvaporationModelBase<number>
  {
  private:
    const RecoilPressureData<number> recoil_data;
    const number                     boiling_temperature;
    const number                     sticking_constant;
    const number                     molar_mass;
    const number                     latent_heat_evaporation;

    // Model constant computed as molar_mass/(2*pi*molar_gas_constant)
    const number Cm;

  public:
    EvaporationModelSaturatedVaporPressure(const RecoilPressureData<number> &recoil_data,
                                           const number                      boiling_temperature,
                                           const number                      molar_mass,
                                           const number latent_heat_evaporation);

    /*
     * The evaporative mass flux is computed as
     *
     *                                    ______
     *                                   /  M
     * .                                /-------
     * m(T) = 0.82 · c_s  · p_sat(T) · √  2πR T
     *
     * where c_s is the sticking coefficient, p_sat(T) is the saturated vapor
     * pressure, M the molar mass, R the molar gas constant and T the temperature.
     */
    number
    local_compute_evaporative_mass_flux(const number T) const final;

    dealii::VectorizedArray<number>
    local_compute_evaporative_mass_flux_vec(const dealii::VectorizedArray<number> &T) const final;

    /**
     * Compute the derivative of the evaporative mass flux as
     *
     *    .
     *  d m     .    /  c_T       1  \
     * ----- =  m(T) | ----- - ----- |
     *  d T          \   T²     2 T  /
     *
     */
    number
    local_compute_evaporative_mass_flux_derivative(const number T) const final;

    dealii::VectorizedArray<number>
    local_compute_evaporative_mass_flux_vec_derivative(
      const dealii::VectorizedArray<number> &T) const final;
  };
} // namespace MeltPoolDG::Evaporation
