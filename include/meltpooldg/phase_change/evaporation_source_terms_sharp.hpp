#pragma once
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_source_terms_base.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   * TODO
   * */
  template <int dim, typename number>
  class EvaporationSourceTermsSharp : public EvaporationSourceTermsBase<dim, number>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;
    const ScratchData<dim, dim, number> &scratch_data;
    const EvaporationData<number>       &evapor_data;
    /**
     * references to solutions needed for the computation
     */
    const VectorType      &level_set_as_heaviside;
    const BlockVectorType &normal_vector;
    const VectorType      &evaporative_mass_flux;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int ls_hanging_nodes_dof_idx;
    const unsigned int ls_quad_idx;
    const unsigned int normal_dof_idx;
    const unsigned int evapor_vel_dof_idx;
    const unsigned int evapor_mass_flux_dof_idx;
    const number       tolerance_normal_vector;
    const number       density_vapor;
    const number       density_liquid;

    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<number> /*weights*/
                                 >> *surface_mesh_info = nullptr;

  public:
    EvaporationSourceTermsSharp(const ScratchData<dim, dim, number> &scratch_data,
                                const EvaporationData<number>       &evapor_data,
                                const VectorType                    &level_set_as_heaviside,
                                const BlockVectorType               &normal_vector,
                                const VectorType                    &evaporative_mass_flux,
                                const unsigned int                   ls_hanging_nodes_dof_idx,
                                const unsigned int                   ls_quad_idx,
                                const unsigned int                   normal_dof_idx,
                                const unsigned int                   evapor_vel_dof_idx,
                                const unsigned int                   evapor_mass_flux_dof_idx,
                                const number                         tolerance_normal_vector,
                                const number                         density_vapor,
                                const number                         density_liquid);

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

    void
    register_surface_mesh(
      const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                   std::vector<Point<dim>> /*quad_points*/,
                                   std::vector<number> /*weights*/
                                   >> &surface_mesh_info);
  };
} // namespace MeltPoolDG::Evaporation
