/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, UIBK/TUM, May 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**
   *                               .
   * For a given evaporation model m(T) fill a DoF-Vector for the
   * evaporative mass flux.
   */
  template <int dim>
  class EvaporationMassFluxOperatorBase
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

  public:
    virtual void
    compute_evaporative_mass_flux(VectorType       &evaporative_mass_flux,
                                  const VectorType &temperature) const = 0;
  };
} // namespace MeltPoolDG::Evaporation
