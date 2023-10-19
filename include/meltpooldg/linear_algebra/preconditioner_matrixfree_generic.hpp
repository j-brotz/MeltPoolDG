/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, UIBK/TUM, December 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_base.hpp>

namespace MeltPoolDG::Preconditioner
{
  using namespace dealii;

  template <int dim, typename OperatorType>
  class PreconditionerMatrixFreeGeneric : public PreconditionerMatrixFreeBase<dim>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<double>;

    const ScratchData<dim> &scratch_data;
    /*
     * select the relevant DoFHandlers
     */
    const unsigned int dof_idx;
    /*
     * type of preconditioner
     */
    const PreconditionerType &preconditioner_type;
    /*
     * matrix-free operator that provides the following public
     * functions
     * - compute_system_matrix_from_matrixfree(TrilinosWrappers::SparseMatrix&)
     * - compute_inverse_diagonal_from_matrixfree(VectorType&)
     */
    const OperatorType &operator_base;

    /*
     * sparsity pattern for preconditioner matrix if needed
     */
    DynamicSparsityPattern dsp;
    /*
     * sparse preconditioner matrix if needed
     */
    TrilinosWrappers::SparseMatrix preconditioner_system_matrix;

  public:
    PreconditionerMatrixFreeGeneric(const ScratchData<dim>   &scratch_data_in,
                                    const unsigned int        curv_dof_idx_in,
                                    const PreconditionerType &preconditioner_type_in,
                                    const OperatorType       &operator_base_in);

    /*
     * setup sparsity pattern of the preconditioner_system_matrix if needed
     */
    void
    reinit() override;

    /*
     * Wrapper to Trilinos preconditioner; supported types: Identity/ILU/AMG
     */
    std::shared_ptr<TrilinosWrappers::PreconditionBase>
    compute_trilinos_preconditioner() override;

    /*
     * compute a diagonal preconditioner
     */
    std::shared_ptr<DiagonalMatrix<VectorType>>
    compute_diagonal_preconditioner() override;

    /*
     * compute a diagonal preconditioner for a block vector
     */
    std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    compute_block_diagonal_preconditioner() override;

    /*----------------- Getter Functions -----------------*/

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const override;

    TrilinosWrappers::SparseMatrix &
    get_system_matrix() override;
  };
} // namespace MeltPoolDG::Preconditioner
