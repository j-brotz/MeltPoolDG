#pragma once

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_mass_flux_operator_base.hpp>
#include <meltpooldg/phase_change/evaporation_model_base.hpp>


namespace MeltPoolDG::Evaporation
{
  /**
   *
   * TODO
   */
  template <int dim, typename number>
  class EvaporationMassFluxOperatorThicknessIntegration
    : public EvaporationMassFluxOperatorBase<dim, number>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    const ScratchData<dim>                                  &scratch_data;
    const EvaporationModelBase<number>                      &evaporation_model;
    const EvaporationData<number>::ThicknessIntegrationData &thickness_integration_data;
    const LevelSet::ReinitializationData<number>            &reinit_data;

    const VectorType      &level_set_as_heaviside;
    const BlockVectorType &normal_vector;

    const unsigned int ls_hanging_nodes_dof_idx;
    const unsigned int normal_dof_idx;
    const unsigned int heat_hanging_nodes_dof_idx;
    const unsigned int evapor_mass_flux_dof_idx;

    const FESystem<dim> fe_dim;

  public:
    EvaporationMassFluxOperatorThicknessIntegration(
      const ScratchData<dim>                                  &scratch_data,
      const EvaporationModelBase<number>                      &evaporation_model,
      const EvaporationData<number>::ThicknessIntegrationData &thickness_integration_data,
      const LevelSet::ReinitializationData<number>            &reinit_data,
      const VectorType                                        &level_set_as_heaviside,
      const BlockVectorType                                   &normal_vector,
      const unsigned int                                       ls_hanging_nodes_dof_idx,
      const unsigned int                                       normal_dof_idx,
      const unsigned int                                       heat_hanging_nodes_dof_idx,
      const unsigned int                                       evapor_mass_flux_dof_idx);

    /**
     *  Compute the evaporative mass flux by means of 1d integration across the
     *  interface thickness as follows
     *
     *  /\ .
     *  |  m δ dx
     *  |     Г
     * \/
     *
     * with the delta function δ .
     *                          Г
     */
    void
    compute_evaporative_mass_flux(VectorType       &evaporative_mass_flux,
                                  const VectorType &temperature) const final;
  };
} // namespace MeltPoolDG::Evaporation
