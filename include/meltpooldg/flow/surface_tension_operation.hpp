/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>
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

    const SurfaceTensionData<double> &data;

    const ScratchData<dim> &scratch_data;

    const VectorType &level_set_as_heaviside;
    const VectorType &solution_curvature;

    const unsigned int ls_dof_idx;
    const unsigned int curv_dof_idx;
    const unsigned int flow_vel_dof_idx;
    const unsigned int flow_pressure_hanging_nodes_dof_idx;
    const unsigned int flow_vel_quad_idx;

    VectorType *      temperature            = nullptr;
    BlockVectorType * solution_normal_vector = nullptr;
    const VectorType *solid                  = nullptr;
    unsigned int      temp_dof_idx;
    unsigned int      normal_dof_idx;
    unsigned int      solid_dof_idx;

    const bool         do_level_set_pressure_gradient_interpolation;
    FullMatrix<double> ls_to_pressure_grad_interpolation_matrix;

    std::unique_ptr<const DeltaApproximationBase<double>> delta_phase_weighted;

  public:
    SurfaceTensionOperation(const SurfaceTensionData<double> &data_in,
                            const ScratchData<dim> &          scratch_data,
                            const VectorType &                level_set_as_heaviside,
                            const VectorType &                solution_curvature,
                            const unsigned int                ls_dof_idx,
                            const unsigned int                curv_dof_idx,
                            const unsigned int                flow_vel_hanging_nodes_dof_idx,
                            const unsigned int                flow_pressure_hanging_nodes_dof_idx,
                            const unsigned int                flow_quad_idx);

    /**
     *  This function introduces the basic framework for temperature-dependent surface tension
     *  forces, i.e. Marangoni convection.
     */
    void
    reinit(const unsigned int temp_dof_idx,
           const unsigned int normal_dof_idx,
           VectorType *       temperature,
           BlockVectorType *  solution_normal_vector);

    void
    reinit(const unsigned int temp_dof_idx,
           const unsigned int normal_dof_idx,
           const unsigned int solid_dof_idx,
           VectorType *       temperature,
           BlockVectorType *  solution_normal_vector,
           const VectorType * solid);

    /*
     *  Compute surface tension
     */
    void
    compute_surface_tension(VectorType &force_rhs, const bool zero_out = true);
  };
} // namespace MeltPoolDG::Flow
