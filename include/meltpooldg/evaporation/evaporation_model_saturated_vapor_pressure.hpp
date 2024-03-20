/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, February 2024
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/evaporation/evaporation_model_base.hpp>
#include <meltpooldg/evaporation/recoil_pressure_data.hpp>

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
  class EvaporationModelSaturatedVaporPressure : public EvaporationModelBase
  {
  private:
    const RecoilPressureData<double> recoil_data;
    const double                     boiling_temperature;
    const double                     sticking_constant;
    const double                     molar_mass;
    const double                     latent_heat_evaporation;

    // Model constant computed as molar_mass/(2*pi*molar_gas_constant)
    const double Cm;

  public:
    EvaporationModelSaturatedVaporPressure(const RecoilPressureData<double> &recoil_data,
                                           const double                      boiling_temperature,
                                           const double                      molar_mass,
                                           const double latent_heat_evaporation);

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
    double
    local_compute_evaporative_mass_flux(const double T) const final;
  };
} // namespace MeltPoolDG::Evaporation
