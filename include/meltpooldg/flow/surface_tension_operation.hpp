/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  /**
   *  This class enables to compute the contribution to interfacial forces due
   *  to temperature-(in)dependent surface tension
   *
   *          /                \        /                         \
   *  f_st =  | N_a,  α κ n  δ |   +    | N_a,  (I - n ⊗ n ) ∇α δ |
   *          \            Γ   /        \             Γ   Γ       /
   *                            Ω                                  Ω
   *
   *  with the temperature-dependent surface tension
   *
   *  α = α_0 - α'_0 ( T - T    )
   *                        α_0
   *
   *  and its gradient
   *
   *  ∇α = -α'_0 ∇T .
   *
   *  @note Since it might happen that α gets negative (which does not make sense)
   *  we compute a minimum value for the surface tension coefficient as
   *
   *          /                                 \
   *  α = max |  α_min, α_0 - α'_0 ( T - T    ) |
   *          \                           α_0   /
   *
   */
  template <int dim>
  class SurfaceTensionOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    //@todo: merge the two following functions

    /*
     *  temperature-independent surface tension
     */
    static void
    compute_surface_tension(VectorType &            force_rhs,
                            const ScratchData<dim> &scratch_data,
                            const VectorType &      level_set_as_heaviside,
                            const VectorType &      curvature_vec,
                            const double            surface_tension_coefficient,
                            const unsigned int      ls_dof_idx,
                            const unsigned int      curv_dof_idx,
                            const unsigned int      flow_vel_hanging_nodes_dof_idx,
                            const unsigned int      flow_quad_idx,
                            const bool              zero_out = true);

    /**
     *  This function introduces the basic framework for temperature-dependent surface tension
     *  forces, i.e. Marangoni convection.
     */
    static void
    compute_temperature_dependent_surface_tension(
      const ScratchData<dim> &scratch_data,
      VectorType &            force_rhs,
      const VectorType &      level_set_as_heaviside,
      const VectorType &      solution_curvature,
      const VectorType &      temperature,
      const BlockVectorType & solution_normal_vector,
      const double            surface_tension_coefficient,
      const double            temperature_dependent_surface_tension_coefficient,
      const double            surface_tension_reference_temperature,
      const double            surface_tension_coefficient_residual_fraction,
      const unsigned int      ls_dof_idx,
      const unsigned int      curv_dof_idx,
      const unsigned int      normal_dof_idx,
      const unsigned int      flow_vel_dof_idx,
      const unsigned int      flow_vel_quad_idx,
      const unsigned int      temp_dof_idx,
      const bool              zero_out = true);
  };
} // namespace MeltPoolDG::Flow
