/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, UIBK/TUM, May 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/phase_change/evaporation_mass_flux_operator_base.hpp>
#include <meltpooldg/phase_change/evaporation_model_base.hpp>

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

    const ScratchData<dim>             &scratch_data;
    const EvaporationModelBase<double> &evaporation_model;

  public:
    EvaporationMassFluxOperatorContinuous(const ScratchData<dim>             &scratch_data,
                                          const EvaporationModelBase<double> &evaporation_model);

    /**
     * DOCU: TODO
     */
    void
    compute_evaporative_mass_flux(VectorType       &evaporative_mass_flux,
                                  const VectorType &temperature) const final;
  };
} // namespace MeltPoolDG::Evaporation
