/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, UIBK/TUM, May 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/evaporation/evaporation_mass_flux_operator_base.hpp>
#include <meltpooldg/evaporation/evaporation_model_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/reinitialization/reinitialization_data.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**
   *
   * TODO
   */
  template <int dim>
  class EvaporationMassFluxOperatorThicknessIntegration
    : public EvaporationMassFluxOperatorBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const ScratchData<dim>                                  &scratch_data;
    const EvaporationModelBase                              &evaporation_model;
    const EvaporationData<double>::ThicknessIntegrationData &thickness_integration_data;
    const LevelSet::ReinitializationData<double>            &reinit_data;

    const VectorType      &level_set_as_heaviside;
    const BlockVectorType &normal_vector;

    const unsigned int ls_hanging_nodes_dof_idx;
    const unsigned int normal_dof_idx;
    const unsigned int temp_hanging_nodes_dof_idx;
    const unsigned int evapor_mass_flux_dof_idx;

    const FESystem<dim> fe_dim;

  public:
    EvaporationMassFluxOperatorThicknessIntegration(
      const ScratchData<dim>                                  &scratch_data,
      const EvaporationModelBase                              &evaporation_model,
      const EvaporationData<double>::ThicknessIntegrationData &thickness_integration_data,
      const LevelSet::ReinitializationData<double>            &reinit_data,
      const VectorType                                        &level_set_as_heaviside,
      const BlockVectorType                                   &normal_vector,
      const unsigned int                                       ls_hanging_nodes_dof_idx,
      const unsigned int                                       normal_dof_idx,
      const unsigned int                                       temp_hanging_nodes_dof_idx,
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
