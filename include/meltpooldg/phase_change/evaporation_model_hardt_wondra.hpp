/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, June 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/phase_change/evaporation_model_base.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   * This class implements the evaporative mass flux as presented in
   *
   * Hardt, S., & Wondra, F. (2008). Evaporation model for interfacial flows
   * based on a continuum-field representation of the source terms.
   * Journal of Computational Physics, 227(11), 5871-5895.
   * DOI: 10.1016/j.jcp.2008.02.020
   *
   * It is derived from the Schrage's theory and incorporates a linear relationship
   * between the evaporation heat or mass ﬂux and the temperature difference at the
   * interface. It is valid in case of small deviations from equilibrium
   *
   * (T_i - T_v)/T_v << 1
   *
   * with T_i the interface temperature and T_v the boiling (saturation) temperature.
   *
   * The evaporative mass transfer coefficient according to Schrage's theory can be
   * computed as
   *
   *            2χ_v    h_v    ρ_v
   * α_v = -----------------------------
   *                   _____        1.5
   *        (2 - χ_v) √2πR_s  ( T_v )
   *
   * where χ_v is the evaporation coefficient, h_v (J/kg) the latent heat of evaporation,
   * ρ_v the vapor density and R_s the specific gas constant (J/(kgK)). Marek and Schraub
   * reported evaporation coefficients to be between 1e−3 and 1.
   *
   */
  template <typename number>
  class EvaporationModelHardtWondra : public EvaporationModelBase<number>
  {
  private:
    const number evaporative_mass_transfer_coefficient;
    const number boiling_temperature;

  public:
    EvaporationModelHardtWondra(const number evaporation_coefficient,
                                const number latent_heat_of_evaporation,
                                const number density_vapor,
                                const number molar_mass_vapor,
                                const number boiling_temperature);

    EvaporationModelHardtWondra(const number evaporative_mass_transfer_coefficient,
                                const number boiling_temperature);

    /*
     * According to Schrage's theory the evaporative mass flux at the interface is computed
     * as follows
     *  .
     *  m = α_v < T - T_v >
     *
     *  with the evaporative mass transfer coefficient α_v  and the heaviside function <...>.
     */
    number
    local_compute_evaporative_mass_flux(const number T) const final;
  };
} // namespace MeltPoolDG::Evaporation
