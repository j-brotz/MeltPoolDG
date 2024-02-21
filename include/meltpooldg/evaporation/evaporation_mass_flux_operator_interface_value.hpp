/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, UIBK/TUM, May 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/evaporation/evaporation_mass_flux_operator_base.hpp>
#include <meltpooldg/evaporation/evaporation_model_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/level_set/level_set_data.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**
   *
   * DOCU: TODO
   */
  template <int dim>
  class EvaporationMassFluxOperatorInterfaceValue : public EvaporationMassFluxOperatorBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const ScratchData<dim>                                     &scratch_data;
    mutable LevelSet::NearestPointData<double>                  nearest_point_data;
    const EvaporationModelBase                                 &evaporation_model;
    const VectorType                                           &level_set_as_heaviside;
    const VectorType                                           &distance;
    const BlockVectorType                                      &normal_vector;
    const unsigned int                                          ls_dof_idx;
    const unsigned int                                          temp_hanging_nodes_dof_idx;
    const unsigned int                                          evapor_mass_flux_dof_idx;
    const double                                                tolerance_normal_vector;
    mutable std::unique_ptr<LevelSet::Tools::NearestPoint<dim>> nearest_point_search;


  public:
    EvaporationMassFluxOperatorInterfaceValue(const ScratchData<dim> &scratch_data,
                                              const LevelSet::NearestPointData<double> &data,
                                              const EvaporationModelBase &evaporation_model,
                                              const VectorType           &level_set_as_heaviside,
                                              const VectorType           &distance,
                                              const BlockVectorType      &normal_vector,
                                              const unsigned int          ls_dof_idx,
                                              const unsigned int temp_hanging_nodes_dof_idx,
                                              const unsigned int evapor_mass_flux_dof_idx);

    /**
     * DOCU: TODO
     */
    void
    compute_evaporative_mass_flux(VectorType       &evaporative_mass_flux,
                                  const VectorType &temperature) const final;
  };
} // namespace MeltPoolDG::Evaporation
