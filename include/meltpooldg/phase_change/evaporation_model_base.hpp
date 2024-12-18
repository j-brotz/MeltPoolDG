/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

namespace MeltPoolDG::Evaporation
{
  /**
   * Base class for implementing different models to compute the evaporative
   * mass flux.
   */
  class EvaporationModelBase
  {
  public:
    /**
     *                                                    .
     * Base function to compute the evaporative mass flux m in kg/(m^2 s) for
     * a given temperature value @p T.
     */
    virtual double
    local_compute_evaporative_mass_flux(const double T) const = 0;
  };
} // namespace MeltPoolDG::Evaporation
