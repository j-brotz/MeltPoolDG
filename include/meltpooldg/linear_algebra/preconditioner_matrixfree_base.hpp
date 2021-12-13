/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, UIBK/TUM, December 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

namespace MeltPoolDG::Preconditioner
{
  using namespace dealii;

  template <int dim>
  class PreconditionerMatrixfreeBase
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

  public:
    virtual void
    reinit() = 0;

    virtual const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const = 0;

    virtual TrilinosWrappers::SparseMatrix &
    get_system_matrix() = 0;

    virtual DiagonalMatrix<VectorType>
    compute_diagonal_preconditioner() = 0;

    virtual std::shared_ptr<TrilinosWrappers::PreconditionBase>
    compute_trilinos_preconditioner() = 0;
  };
} // namespace MeltPoolDG::Preconditioner
