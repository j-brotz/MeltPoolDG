/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

namespace MeltPoolDG::Evaporation
{
  class EvaporationModelBase
  {
  public:
    virtual double
    local_compute_evaporative_mass_flux(const double T) = 0;
  };
} // namespace MeltPoolDG::Evaporation
