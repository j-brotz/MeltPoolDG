#pragma once

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

using namespace dealii;

namespace MeltPoolDG::RadiativeTransport
{
  // clang-format off
  /*
   *  This operator computes the LHS and RHS of the radiative transport equation as described by Lin et al (2020):
   *      ∇ · (s I) + µ_A I = 0
   *      where:  s (/) is the unit direction vector of the laser.
   *              I (W/m^2) is the intensity of the laser. Is a field quantity.
   *              µ_A (/) is the absorption opacity of the surface. It can be based on material constants or heaviside gradients.
   *                In the (default) gradient-based absorption opacity,
   *                it is defined as a function of the heaviside function gradient and the laser direction:
   *                                      1                 /               1            \
   *                µ_A = < ∇H   s   ------------ >   = max | ∇H   s   ------------  , 0 |
   *                                  (1.- H + ϵ)           \          (1.- H + ϵ)       /
   *                      where ϵ is a small scalar constant to avoid a division by zero (avoid_div_zero_constant)
   *                      and <*> represent MacAulay brackets.
   *              
   *  The RTE cast into a weak formulation (Φ is the test function) reads as follows:
   *  ( Φ , ∇ · (s I)  ) + ( Φ ,  µ_A I ) = 0
   *                    Ω                Ω
   *  The divergence term is rearranged to:
   *      ∇ · (s I) = s (∇ · I) + I (∇ · s) = s · ∇I = s*grad(I)
   *  Hence:
   *     /                                                         \
   *     | Φ, ( s · ∇I   +                µ_A        ·        I   )|  = 0
   *     \                                                         /
   *                                                                Ω
   *  or with the full (default) gradient-based absorption opacity definition:
   *     /                                      1                   \
   *     | Φ, ( s · ∇I   +     < ∇H · s · ------------ >    ·  I  ) |  = 0
   *     \                                 (1.- H + ϵ)              /
   *                                                                Ω
   */
  // clang-format on
  template <int dim, typename number = double>
  class RadiativeTransportOperator : public OperatorBase<dim, number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using OperatorBase<dim, number>::vmult;
    using OperatorBase<dim, number>::assemble_matrixbased;
    using OperatorBase<dim, number>::create_rhs;
    using OperatorBase<dim, number>::compute_inverse_diagonal_from_matrixfree;

  private:
    using VectorType       = LinearAlgebra::distributed::Vector<number>;
    using SparseMatrixType = TrilinosWrappers::SparseMatrix;
    using vector           = Tensor<1, dim, VectorizedArray<number>>;
    using scalar           = VectorizedArray<number>;

    const ScratchData<dim> &scratch_data;

    const RadiativeTransportData<double> &rte_data;

    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    Tensor<1, dim, number> laser_direction;

    const double pure_gas_level_set;
    const double pure_liquid_level_set;

  public:
    RadiativeTransportOperator(const ScratchData<dim>               &scratch_data_in,
                               const RadiativeTransportData<double> &rte_data,
                               const VectorType                     &heaviside_in,
                               const unsigned int                    rte_dof_idx_in,
                               const unsigned int                    rte_quad_idx_in,
                               const unsigned int                    hs_dof_idx_in);

    /*
     *  matrix-free utility
     */

    void
    vmult(VectorType &dst, const VectorType &src) const final;

    void
    create_rhs(VectorType &dst, const VectorType &src) const final;

    void
    compute_system_matrix_from_matrixfree(
      TrilinosWrappers::SparseMatrix &system_matrix) const final;

    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

  private:
    void
    tangent_local_cell_operation(FECellIntegrator<dim, 1, number> &intensity_vals,
                                 FECellIntegrator<dim, 1, number> &level_set_vals,
                                 const bool                        do_reinit_cells) const;
  };

  template <int dim, typename number = double>
  inline VectorizedArray<number>
  compute_mu(const RadiativeTransportData<number>          &rte_data,
             const VectorizedArray<number>                 &H,
             const Tensor<1, dim, VectorizedArray<number>> &grad_H,
             const Tensor<1, dim, number>                  &laser_direction,
             const double                                   avoid_div_zero_constant)
  {
    // 1. material constant mu
    if (rte_data.absorptivity_type == AbsorptivityType::constant)
      return (
        LevelSet::Tools::interpolate(H,
                                     rte_data.absorptivity_constant_data.absorptivity_gas,
                                     rte_data.absorptivity_constant_data.absorptivity_liquid));

    // 2. gradient based mu : max(0, ∇H * laser_dir *1./(1.- H + ϵ))
    else if (rte_data.absorptivity_type == AbsorptivityType::gradient_based)
      {
        VectorizedArray<number> dummy =
          scalar_product(grad_H, laser_direction) * 1. / (1. - H + avoid_div_zero_constant);
        return compare_and_apply_mask<SIMDComparison::less_than>(dummy,
                                                                 0.,
                                                                 /*true*/ 0.,
                                                                 /*false*/ dummy);
      }

    else
      AssertThrow(false, ExcNotImplemented());
    return VectorizedArray<number>(0);
  }


  /**
   * This function returns a mask to identify potentially singular matrix blocks characterized by
   * values equal to zero. Due to the mathematical forumulation of the radiative transfer equation,
   * this may arise if the intensity and the intensity gradient is zero in pure liquid or pure
   * gaseous state.
   *
   * @param I Intensity values.
   * @param grad_I Intensity gradients.
   * @param H Heaviside values.
   * @param pure_liquid_level_set Upper threshold for heaviside values of a pure liquid state.
   */
  template <int dim, typename number = double>
  inline VectorizedArray<number>
  compute_invalid_mask(const VectorizedArray<number>                 &I,
                       const Tensor<1, dim, VectorizedArray<number>> &grad_I,
                       const VectorizedArray<number>                 &H,
                       const double                                   pure_liquid_level_set)
  {
    // only invalid mask on pure liquid
    VectorizedArray<number> liquid_checker =
      compare_and_apply_mask<SIMDComparison::greater_than>(H, pure_liquid_level_set, 1., 0.);
    if (liquid_checker.sum() > (H.size() - 1))
      return VectorizedArray<number>(1.);

    VectorizedArray<number> intensity_grad_magnitude = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
      intensity_grad_magnitude += std::abs(grad_I[d]);

    // else, if cell is in the gas phase H=0, following conditions needs to be met:
    //   cell must experience no intensity
    //   gradient of intensity AND heaviside IN ANY DIRECTION must be zero
    // this makes sure that invalid_mask avoids the intensity column entirely, even on sides
    return compare_and_apply_mask<SIMDComparison::greater_than>(
      compare_and_apply_mask<SIMDComparison::less_than>(I, VectorizedArray<double>(1e-16), 1., 0.) +
        compare_and_apply_mask<SIMDComparison::greater_than>(
          intensity_grad_magnitude, VectorizedArray<double>(1e-16), 1., 0.),
      2. - 1e-16,
      1.,
      0.);
  }
} // namespace MeltPoolDG::RadiativeTransport
