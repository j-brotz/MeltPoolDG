/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, June 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/evaporation/evaporation_model_base.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   * This class implements the evaporative mass flux computed from the
   * recoil pressure as presented in, e.g.,
   *
   * S.A. Khairallah, A.T. Anderson, A. Rubenchik, W.E. King,
   * Laser powder-bed fusion additive manufacturing: Physics of complex
   * melt flow and formation mechanisms of pores, spatter, and denudation
   * zones, Acta Mater. 108 (2016) 36–45.
   * DOI: 10.1016/j.actamat.2016.02.014.
   *
   */
  template <int dim>
  class EvaporationModelRecoilPressure : public EvaporationModelBase
  {
  private:
    const double boiling_temperature;
    const double pressure_constant;
    const double temperature_constant;
    const double mass_flux_scale_factor;

    // according to Meier 2020
    const double sticking_constant;
    const double Cm; // molar_mass/(2*pi*molar_gas_constant)

  public:
    EvaporationModelRecoilPressure(const double boiling_temperature,
                                   const double pressure_constant,
                                   const double temperature_constant,
                                   const double sticking_constant,
                                   const double molar_mass,
                                   const double mass_flux_scale_factor = 1.0);

    /*
     * The evaporative mass flux is computed as
     *
     *                                  ______
     *                                 /  M
     * .                              /-------
     * m(T) = 0.82 · c_s  · p_r(T) · √  2πR T
     *
     * where c_s is the sticking coefficient, p_r(T) the recoil pressure,
     * M the molar mass, R the molar gas constant and T the temperature.
     */
    double
    local_compute_evaporative_mass_flux(const double T) const final;
  };
} // namespace MeltPoolDG::Evaporation
