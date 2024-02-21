/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, UIBK/TUM, April 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/heat_transfer_operator.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_base.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  //@todo: It could be inherit from PreconditionerGeneric to save some code duplicates
  template <int dim>
  class HeatTransferPreconditionerMatrixFree
    : public Preconditioner::PreconditionerMatrixFreeBase<dim>
  {
  private:
    using VectorType   = LinearAlgebra::distributed::Vector<double>;
    using OperatorType = std::shared_ptr<HeatTransferOperator<dim>>;

    const ScratchData<dim> &scratch_data;
    /**
     * select the relevant DoFHandlers
     */
    const unsigned int temp_dof_idx;
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
    const OperatorType &heat_operator;
    /*
     * base name of preconditioner (e.g. AMG if preconditioner_type=AMG_reduced)
     */
    PreconditionerType preconditioner_base_name = PreconditionerType::Identity;
    /*
     * sparsity pattern for preconditioner matrix if needed
     */
    DynamicSparsityPattern dsp;
    /*
     * sparse preconditioner matrix if needed
     */
    TrilinosWrappers::SparseMatrix preconditioner_system_matrix;

  public:
    HeatTransferPreconditionerMatrixFree(const ScratchData<dim>   &scratch_data_in,
                                         unsigned int              temp_dof_idx_in,
                                         const PreconditionerType &preconditioner_type_in,
                                         const OperatorType       &operator_base_in);

    void
    reinit() override;

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const override;

    TrilinosWrappers::SparseMatrix &
    get_system_matrix() override;

    std::shared_ptr<DiagonalMatrix<VectorType>>
    compute_diagonal_preconditioner() override;

    std::shared_ptr<TrilinosWrappers::PreconditionBase>
    compute_trilinos_preconditioner() override;
  };
} // namespace MeltPoolDG::Heat
