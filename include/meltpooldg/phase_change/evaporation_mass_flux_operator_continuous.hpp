#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/phase_change/evaporation_mass_flux_operator_base.hpp>
#include <meltpooldg/phase_change/evaporation_model_base.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   *                               .
   * DOCU: TODO
   */
  template <int dim, typename number>
  class EvaporationMassFluxOperatorContinuous : public EvaporationMassFluxOperatorBase<dim, number>
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    const ScratchData<dim, dim, number> &scratch_data;
    const EvaporationModelBase<number>  &evaporation_model;

  public:
    EvaporationMassFluxOperatorContinuous(const ScratchData<dim, dim, number> &scratch_data,
                                          const EvaporationModelBase<number>  &evaporation_model);

    /**
     * DOCU: TODO
     */
    void
    compute_evaporative_mass_flux(VectorType       &evaporative_mass_flux,
                                  const VectorType &temperature) const final;
  };
} // namespace MeltPoolDG::Evaporation
