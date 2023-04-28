#pragma once

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>

using namespace dealii;

namespace MeltPoolDG::RadiativeTransport
{
  // clang-format off
/*
 *  This operator computes the LHS and RHS of the radiative transport equation as described by Lin et al (2020):
 *      ∇ · (s I) + µ_A I = 0
 *      where:  s (/) is the unit direction vector of the laser.
 *                  Taken to be constant and straight downwards facing, it is (0.0,-1.0) in 2D and (0.0,0.0,-1.0) in 3D
 *              I (W/m^2) is the intensity of the laser. Is a field quantity
 *              µ_A (/) is the absorption opacity of the surface.
 *                  It is described as a heaviside function, whose scale and offset is given as input parameters
 *
 *  Cast into a weak formulation (Φ is the test function) :
 *  ( Φ , ∇ · (s I)  ) + ( Φ ,  µ_A I ) = 0
 *                    Ω                Ω
 *  Because s  is taken to be constant, integration by parts is not necessary:
 *      ∇ · (s I) = s (∇ · I) + I (∇ · s) = s∇I = s*grad(I)
 *  Hence:
 *     /                        \
 *  ∫ Φ|  ( s  ∇I )  +    µ_A*I |  = 0
 *     \                        /
 *                              Ω
 *
 *                                              */


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

    const VectorType &intensity;
    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    Tensor<1, dim, number> laser_direction;

  public:
    RadiativeTransportOperator(const ScratchData<dim> &              scratch_data_in,
                               const RadiativeTransportData<double> &rte_data,
                               VectorType &                          intensity_in,
                               const VectorType &                    heaviside_in,
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

    void
    reinit() final;

  private:
    void
    tangent_local_cell_operation(FECellIntegrator<dim, 1, number> &intensity_vals,
                                 FECellIntegrator<dim, 1, number> &level_set_vals,
                                 const bool                        do_reinit_cells) const;
  };
} // namespace MeltPoolDG::RadiativeTransport
