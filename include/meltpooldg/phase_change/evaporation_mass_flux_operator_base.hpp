#pragma once

#include <deal.II/lac/la_parallel_vector.h>

namespace MeltPoolDG::Evaporation
{
  /**
   *                               .
   * For a given evaporation model m(T) fill a DoF-Vector for the
   * evaporative mass flux.
   */
  template <int dim, typename number>
  class EvaporationMassFluxOperatorBase
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    virtual void
    compute_evaporative_mass_flux(VectorType       &evaporative_mass_flux,
                                  const VectorType &temperature) const = 0;
  };
} // namespace MeltPoolDG::Evaporation
