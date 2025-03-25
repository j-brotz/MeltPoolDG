#pragma once
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_source_terms_base.hpp>
#include <meltpooldg/utilities/material_data.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   * TODO: DOCU
   */
  template <int dim, typename number>
  class EvaporationSourceTermsContinuous : public EvaporationSourceTermsBase<dim, number>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;
    const ScratchData<dim>        &scratch_data;
    const EvaporationData<number> &evapor_data;
    /**
     * references to solutions needed for the computation
     */
    const VectorType      &level_set_as_heaviside;
    const BlockVectorType &normal_vector;
    const VectorType      &evaporative_mass_flux;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int                          ls_hanging_nodes_dof_idx;
    const unsigned int                          ls_quad_idx;
    const unsigned int                          normal_dof_idx;
    const unsigned int                          evapor_vel_dof_idx;
    const unsigned int                          evapor_mass_flux_dof_idx;
    const number                                tolerance_normal_vector;
    const number                                density_vapor;
    const number                                density_liquid;
    const TwoPhaseFluidPropertiesTransitionType two_phase_properties_transition_type;

    // interpolation matrix to interpolate the level set gradient to pressure space
    dealii::FullMatrix<number> ls_to_pressure_grad_interpolation_matrix;
    /**
     * evaporation velocity at quadrature points
     */
    dealii::AlignedVector<Tensor<1, dim, dealii::VectorizedArray<number>>> evaporation_velocities;

    inline dealii::Tensor<1, dim, dealii::VectorizedArray<number>> *
    begin_evaporation_velocity(const unsigned int macro_cell);

    inline const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &
    begin_evaporation_velocity(const unsigned int macro_cell) const;

  public:
    EvaporationSourceTermsContinuous(
      const ScratchData<dim>                      &scratch_data,
      const EvaporationData<number>               &evapor_data,
      const VectorType                            &level_set_as_heaviside,
      const BlockVectorType                       &normal_vector,
      const VectorType                            &evaporative_mass_flux,
      const unsigned int                           ls_hanging_nodes_dof_idx,
      const unsigned int                           ls_quad_idx,
      const unsigned int                           normal_dof_idx,
      const unsigned int                           evapor_vel_dof_idx,
      const unsigned int                           evapor_mass_flux_dof_idx,
      const number                                 tolerance_normal_vector,
      const number                                 density_vapor,
      const number                                 density_liquid,
      const TwoPhaseFluidPropertiesTransitionType &two_phase_properties_transition_type);

    void
    compute_evaporation_velocity(VectorType &evaporation_velocity) final;

    void
    compute_level_set_source_term(VectorType        &level_set_source_term,
                                  const unsigned int ls_dof_idx,
                                  const VectorType  &level_set,
                                  const unsigned int pressure_dof_idx) final;

    void
    compute_mass_balance_source_term(VectorType        &mass_balance_source_term,
                                     const unsigned int pressure_dof_idx,
                                     const unsigned int pressure_quad_idx,
                                     bool               zero_out) final;

    void
    compute_heat_source_term(VectorType &heat_source_term) final;
  };
} // namespace MeltPoolDG::Evaporation
