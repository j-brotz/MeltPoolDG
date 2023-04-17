#pragma once

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>

using namespace dealii;

namespace MeltPoolDG::RadiativeTransport
{
  /**
   * TODO
   *
   */
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
    // using scalar              = VectorizedArray<number>;

    const ScratchData<dim> &scratch_data;

    const RadiativeTransportData<double> &rte_data;

    const VectorType &intensity;
    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

  public:
    RadiativeTransportOperator(const ScratchData<dim> &              scratch_data_in,
                               const RadiativeTransportData<double> &rte_data,
                               VectorType &                          intensity_in,
                               const VectorType &                    heaviside_in,
                               const unsigned int                    rte_dof_idx_in,
                               const unsigned int                    rte_quad_idx_in,
                               const unsigned int                    hs_dof_idx_in);

    void
    assemble_matrixbased(const VectorType &intensity_old,
                         SparseMatrixType &matrix,
                         VectorType &      rhs) const final;

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
