/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, June 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/evaporation/evaporation_source_terms_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**
   * TODO
   * */
  template <int dim>
  class EvaporationSourceTermsSharp : public EvaporationSourceTermsBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;
    const ScratchData<dim>        &scratch_data;
    const EvaporationData<double> &evapor_data;
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
    const double       tolerance_normal_vector;
    const double       density_vapor;
    const double       density_liquid;

    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<double> /*weights*/
                                 >> *surface_mesh_info = nullptr;

  public:
    EvaporationSourceTermsSharp(const ScratchData<dim>        &scratch_data,
                                const EvaporationData<double> &evapor_data,
                                const VectorType              &level_set_as_heaviside,
                                const BlockVectorType         &normal_vector,
                                const VectorType              &evaporative_mass_flux,
                                const unsigned int             ls_hanging_nodes_dof_idx,
                                const unsigned int             ls_quad_idx,
                                const unsigned int             normal_dof_idx,
                                const unsigned int             evapor_vel_dof_idx,
                                const unsigned int             evapor_mass_flux_dof_idx,
                                const double                   tolerance_normal_vector,
                                const double                   density_vapor,
                                const double                   density_liquid);

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
                                   std::vector<double> /*weights*/
                                   >> &surface_mesh_info);
  };
} // namespace MeltPoolDG::Evaporation
