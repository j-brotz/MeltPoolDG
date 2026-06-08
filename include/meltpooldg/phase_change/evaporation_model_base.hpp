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
    virtual ~EvaporationModelBase() = default;
    /**
     *                                                    .
     * Base function to compute the evaporative mass flux m in kg/(m^2 s) for
     * a given temperature value @p T.
     */
    virtual number
    local_compute_evaporative_mass_flux(const number T) const = 0;
    /**
     *                                                    .
     * Base function to compute the evaporative mass flux m in kg/(m^2 s) for
     * a given temperature value @p T.
     */
    virtual dealii::VectorizedArray<number>
    local_compute_evaporative_mass_flux_vec(const dealii::VectorizedArray<number> & /*T*/) const
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

    /**
     * Compute the derivative of the evaporative mass flux.
     */
    virtual number
    local_compute_evaporative_mass_flux_derivative(const number /*T*/) const
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

    virtual dealii::VectorizedArray<number>
    local_compute_evaporative_mass_flux_vec_derivative(
      const dealii::VectorizedArray<number> & /*T*/) const
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  };
} // namespace MeltPoolDG::Evaporation
