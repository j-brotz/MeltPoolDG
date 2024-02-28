/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/flow/surface_tension_data.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>

#include <memory>

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
   *                           Ω                                  Ω
   *
   *  with the curvature
   *
   *  κ = -( ∇ ∙ n )
   *
   *  the temperature-dependent surface tension coefficient
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

    const VectorType      *temperature            = nullptr;
    const BlockVectorType *solution_normal_vector = nullptr;
    const VectorType      *solid                  = nullptr;
    unsigned int           temp_dof_idx;
    unsigned int           normal_dof_idx;
    unsigned int           solid_dof_idx;

    const bool         do_level_set_pressure_gradient_interpolation;
    FullMatrix<double> ls_to_pressure_grad_interpolation_matrix;

    std::unique_ptr<const LevelSet::DeltaApproximationBase<double>> delta_phase_weighted;

    const double alpha_residual;

  public:
    SurfaceTensionOperation(const SurfaceTensionData<double> &data_in,
                            const ScratchData<dim>           &scratch_data,
                            const VectorType                 &level_set_as_heaviside,
                            const VectorType                 &solution_curvature,
                            const unsigned int                ls_dof_idx,
                            const unsigned int                curv_dof_idx,
                            const unsigned int                flow_vel_hanging_nodes_dof_idx,
                            const unsigned int                flow_pressure_hanging_nodes_dof_idx,
                            const unsigned int                flow_quad_idx);

    /**
     * Registers the DoF vectors of the @p temperature and normal vector (@p solution_normal_vector),
     * which are required for temperature-dependent surface tension forces, i.e. Marangoni
     * convection.
     */
    void
    register_temperature_and_normal_vector(const unsigned int     temp_dof_idx,
                                           const unsigned int     normal_dof_idx,
                                           const VectorType      *temperature,
                                           const BlockVectorType *solution_normal_vector);
    /**
     * Registers the solid fraction, given by a DoF vector @p solid and a DoF index @p solid_dof_idx,
     * which is required if surface tension should be zeroed out in the solid domain.
     */
    void
    register_solid_fraction(const unsigned int solid_dof_idx, const VectorType *solid);

    /**
     * Compute a DoF vector of the surface tension and add it into to @p force_rhs. If the
     * latter should be zeroed out first, @p zero_out must be set true.
     */
    void
    compute_surface_tension(VectorType &force_rhs, const bool zero_out = true);

    /**
     * Compute the time step limit for explicit treatment of surface tension [*], as
     *
     *                __________________
     *               / (ρ1 + ρ2) * Δx^3
     *              / ------------------
     * Δt   = k   \/       2 π α
     *   st    st
     *
     * with an arbitrary scale factor 0 <= k  <= 1, the densities of the two fluids, ρ1 (@p
     * density_1) st and ρ2 (@p density_2), the element size Δx (minimal edge length) and the
     * surface tension coefficient α.
     *
     * [*] J.U. Brackbill, D.B. Kothe, C. Zemach: A continuum method for modeling surface
     *      tension, J. Comput. Phys. 100 (2) (1992) 335–354.
     *
     * @note As a conservative assumption, we choose α as max (α (T)) in case of
     * temperature-dependent surface tension.
     */
    double
    compute_time_step_limit(const double density_1, const double density_2);

  private:
    /**
     * Compute the temperature-dependent surface tension coefficient for a given temperature
     * @p T, according to
     *
     *  α = α_0 - α'_0 ( T - T    ).
     *                        α_0
     *
     * @note If α'_0 is positive, α decreases with increasing temperature.
     */
    template <typename number>
    number
    local_compute_temperature_dependent_surface_tension_coefficient(const number &T);
  };
} // namespace MeltPoolDG::Flow
