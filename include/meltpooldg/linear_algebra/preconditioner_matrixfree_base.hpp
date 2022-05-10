/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, UIBK/TUM, December 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

namespace MeltPoolDG::Preconditioner
{
  using namespace dealii;

  template <int dim>
  class PreconditionerMatrixFreeBase
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<double>;

  public:
    virtual void
    reinit() = 0;

    virtual const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const = 0;

    virtual TrilinosWrappers::SparseMatrix &
    get_system_matrix() = 0;

    virtual std::shared_ptr<DiagonalMatrix<VectorType>>
    compute_diagonal_preconditioner() = 0;

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    compute_block_diagonal_preconditioner()
    {
      AssertThrow(false, ExcNotImplemented());
      return nullptr;
    }

    virtual std::shared_ptr<TrilinosWrappers::PreconditionBase>
    compute_trilinos_preconditioner() = 0;
  };
} // namespace MeltPoolDG::Preconditioner
