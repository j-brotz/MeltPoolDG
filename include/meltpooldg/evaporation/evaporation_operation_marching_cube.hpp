/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/utilities/vector_tools.hpp>

#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <adaflo/sharp_interface_util.h> //@todo: will be replace by the utility function of deal.II as soon the PR is merged
#endif

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**                                                        .
   *  This module computes for a given evaporative mass flux m the corresponding
   *  term in the continuity equation in a sharp manner exploiting the Marching
   *  Cube Algorithm.
   *
   *  /           \     /    .    1       1    \
   *  | w , ∇ · u |   = | w, m ( ---  -  --- ) |
   *  \           /     \         ρl      ρg   /
   *              Ω                             Γ
   *                                             lg
   *  with the domain of interest Ω, the test functions w, the liquid-gaseous
   *  interface Γ_lg, the fluid velocity field u, density of the liquid phase ρl
   *  and density of the gaseous phase ρg.
   *
   *  The evaporation velocity is then computed as follows
   *
   *          /  .
   *          |  m / ρl    if phi > 0
   *  u = n   |  .
   *       Γ  |  m / ρg    else
   *          \
   */
  template <int dim>
  class EvaporationOperationMarchingCube
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    static void
    compute_evaporation_velocity(const ScratchData<dim> &scratch_data,
                                 VectorType &            evaporation_velocity,
                                 const VectorType &      evaporative_mass_flux,
                                 const VectorType &      level_set_as_heaviside,
                                 const BlockVectorType & normal_vector,
                                 const double            rho_l,
                                 const double            rho_g,
                                 const unsigned int      evapor_vel_dof_idx,
                                 const unsigned int      ls_hanging_nodes_dof_idx,
                                 const unsigned int      ls_quad_idx,
                                 const unsigned int      normal_dof_idx);

    static void
    compute_mass_balance_source_term_sharp(const ScratchData<dim> &scratch_data,
                                           VectorType &            mass_balance_rhs,
                                           const VectorType &      evaporative_mass_flux,
                                           const VectorType &      level_set_vector,
                                           const double            rho_l,
                                           const double            rho_g,
                                           const unsigned int      evapor_dof_idx,
                                           const unsigned int      pressure_dof_idx);
  };
} // namespace MeltPoolDG::Evaporation
