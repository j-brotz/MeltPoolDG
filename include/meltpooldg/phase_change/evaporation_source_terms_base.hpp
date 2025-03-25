#pragma once
#include <deal.II/lac/la_parallel_vector.h>

namespace MeltPoolDG::Evaporation
{
  /**
   * Base class for implementing different ways to compute the source terms for
   *    -- the level set equation
   *    -- the mass balance equation
   *    -- the heat equation
   */
  template <int dim, typename number>
  class EvaporationSourceTermsBase
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    virtual ~EvaporationSourceTermsBase() = default;

    virtual void
    compute_evaporation_velocity(VectorType &evaporation_velocity) = 0;

    virtual void
    compute_level_set_source_term(VectorType        &level_set_source_term,
                                  const unsigned int ls_dof_idx,
                                  const VectorType  &level_set,
                                  const unsigned int pressure_dof_idx) = 0;

    virtual void
    compute_mass_balance_source_term(VectorType        &mass_balance_source_term,
                                     const unsigned int pressure_dof_idx,
                                     const unsigned int pressure_quad_idx,
                                     bool               zero_out) = 0;

    virtual void
    compute_heat_source_term(VectorType &heat_source_term) = 0;
  };
} // namespace MeltPoolDG::Evaporation
