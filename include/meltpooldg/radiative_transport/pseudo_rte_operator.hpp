#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>


using namespace dealii;

namespace MeltPoolDG::RadiativeTransport
{
  // clang-format off
  /*
   *  This operator solves for a (pseudo-)time dependent problem
   *  of the Radiative Transport Equation (RTE) problem with an optional diffusion term.
   *  It can be used to solve for a predictor to the plain RTE problem,
   *              or to solve fully (pseudo-)time dependent problem.
   *
   *  The semi-discrete pseudo-time dependent RTE problem is:
   *  I^(n+1) = I^(n) + dt * [ - ∇ · (s I^(n)) - µ_A I^(n) + D ΔI^(n) ]
   *      where:  s (/) is the unit direction vector of the laser.
   *              I (W/m^2) is the intensity of the laser. Is a field quantity.
   *              D (/) is the diffusion term scaling, or intensity diffusivity coefficient.
   *                It is optional, not physical and is only to create numerical diffusion.
   *              µ_A (/) is the absorption opacity of the surface. It can be based on material constants or heaviside gradients.
   *                In the (default) gradient-based absorption opacity,
   *                it is defined as a function of the heaviside function gradient and the laser direction:
   *                                      1                 /               1            \
   *                µ_A = < ∇H   s   ------------ >   = max | ∇H   s   ------------  , 0 |
   *                                  (1.- H + ϵ)           \          (1.- H + ϵ)       /
   *                      where ϵ is a small scalar constant to avoid a division by zero (avoid_div_zero_constant)
   *                      and <*> represent MacAulay brackets.
   *  The pseudo-RTE cast into a weak formulation (Φ is the test function) reads as follows:
   *  ( Φ , I^(n+1)  )    = ( Φ ,  I^(n) + dt * [ - ∇ · (s I^(n)) - µ_A · I^(n) + D · ΔI^(n) ]   )
   *                  Ω                                                                          Ω
   *  The divergence term is rearranged to:
   *      ∇ · (s I) = s (∇ · I) + I (∇ · s) = s · ∇I = s*grad(I)
   *  Hence:
   *  /           \   /                                                                                   \
   *  | Φ, I^(n+1)| = | Φ, I^(n) + dt * (-s · ∇I^(n) -             µ_A             ·  I^(n) + D · ΔI^(n) )|
   *  \           /   \                                                                                   /
   *              Ω                                                                                       Ω
   *  or with the full (default) gradient-based  absorption opacity definition:
   *  /           \   /                                                1                                  \
   *  | Φ, I^(n+1)| = | Φ, I^(n) + dt * (-s · ∇I^(n) - < ∇H · s · ------------ >    · I^(n) + D · ΔI^(n) )|
   *  \           /   \                                           (1.- H + ϵ)                             /
   *              Ω
   */
  // clang-format on


  template <int dim, typename number = double>
  class PseudoRTEOperator : public OperatorBase<dim, number>
  {
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

    const Tensor<1, dim, number> &laser_direction;

    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    const double pure_gas_level_set;
    const double pure_liquid_level_set;

  public:
    PseudoRTEOperator(const ScratchData<dim>               &scratch_data_in,
                      const RadiativeTransportData<double> &rte_data_in,
                      const Tensor<1, dim, number>         &laser_direction_in,
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
    tangent_local_cell_operation(FECellIntegrator<dim, 1, number> &intensity_vals) const;
  };
} // namespace MeltPoolDG::RadiativeTransport
