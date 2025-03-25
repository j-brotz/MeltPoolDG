#pragma once

#include <deal.II/base/vectorization.h>

namespace MeltPoolDG::Evaporation
{
  /**
   * Base class for implementing different models to compute the evaporative
   * mass flux.
   */
  template <typename number>
  class EvaporationModelBase
  {
  public:
    /**
     *                                                    .
     * Base function to compute the evaporative mass flux m in kg/(m^2 s) for
     * a given temperature value @p T.
     */
    virtual number
    local_compute_evaporative_mass_flux(const number T) const = 0;
  };
} // namespace MeltPoolDG::Evaporation
