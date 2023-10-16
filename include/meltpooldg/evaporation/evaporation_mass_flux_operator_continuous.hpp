/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, UIBK/TUM, May 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/evaporation/evaporation_mass_flux_operator_base.hpp>
#include <meltpooldg/evaporation/evaporation_model_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**
   *                               .
   * DOCU: TODO
   */
  template <int dim>
  class EvaporationMassFluxOperatorContinuous : public EvaporationMassFluxOperatorBase<dim>
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const ScratchData<dim>     &scratch_data;
    const EvaporationModelBase &evaporation_model;

  public:
    EvaporationMassFluxOperatorContinuous(const ScratchData<dim>     &scratch_data,
                                          const EvaporationModelBase &evaporation_model);

    /**
     * DOCU: TODO
     */
    void
    compute_evaporative_mass_flux(VectorType       &evaporative_mass_flux,
                                  const VectorType &temperature) const final;
  };
} // namespace MeltPoolDG::Evaporation
