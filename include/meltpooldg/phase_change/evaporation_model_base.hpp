/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/base/vectorization.h>

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

    virtual dealii::VectorizedArray<double>
    local_compute_evaporative_mass_flux(const dealii::VectorizedArray<double> &T) const
    {
      // default implementation
      dealii::VectorizedArray<double> out;
      for (unsigned int i = 0; i < dealii::VectorizedArray<double>::size(); ++i)
        out[i] = this->local_compute_evaporative_mass_flux(T[i]);
      return out;
    };
  };
} // namespace MeltPoolDG::Evaporation
