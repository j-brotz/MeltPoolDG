/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, UIBK/TUM, December 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_base.hpp>

#include <memory>
#include <variant>

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
    /**
     * @brief Type alias representing a variant type for different preconditioner objects.
     *
     * This variant type can hold either a shared pointer to a Trilinos-based preconditioner
     * or a diagonal matrix-based preconditioner.
     */
    using PreconditionerObjectType =
      std::variant<std::shared_ptr<TrilinosWrappers::PreconditionBase>,
                   std::shared_ptr<DiagonalMatrix<VectorType>>>;

    /**
     * @brief Constructs a preconditioner with matrix-free settings and specified parameters.
     *
     * @param scratch_data_in Reference to the ScratchData object for matrix-free operations.
     * @param dof_idx_in Index of the current DoF layout.
     * @param preconditioner_type_in The type of preconditioner to use (e.g., Diagonal, ILU, AMG).
     * @param operator_base_in Reference to the operator providing necessary functionality.
     */
    PreconditionerMatrixFreeGeneric(const ScratchData<dim>   &scratch_data_in,
                                    const unsigned int        dof_idx_in,
                                    const PreconditionerType &preconditioner_type_in,
                                    const OperatorType       &operator_base_in);

    /**
     * @brief Sets up the sparsity pattern of the preconditioner system matrix if required.
     */
    void
    reinit() override;

    /**
     * @brief Creates a Trilinos-based preconditioner.
     *
     * Supported types include Identity, ILU, and AMG. This function serves as a wrapper
     * for Trilinos preconditioner creation.
     *
     * @return A shared pointer to a TrilinosWrappers::PreconditionBase object.
     */

    std::shared_ptr<TrilinosWrappers::PreconditionBase>
    compute_trilinos_preconditioner() override;

    /**
     * @brief Computes a diagonal preconditioner.
     *
     * Constructs a diagonal preconditioner useful for matrix-free solver contexts.
     *
     * @return A shared pointer to a DiagonalMatrix<VectorType> object.
     */
    std::shared_ptr<DiagonalMatrix<VectorType>>
    compute_diagonal_preconditioner() override;

    /**
     * @brief Computes and returns the selected preconditioner.
     *
     * This method chooses and constructs the appropriate preconditioner based on
     * internal configuration. It returns a variant type encapsulating either a
     * Trilinos-based or diagonal preconditioner.
     *
     * @return A PreconditionerObjectType containing the selected preconditioner.
     *
     * @note use std::visit() to pass the PreconditionerObjectType to a function e.g.
     * via
     * std::visit(
     *   [&](auto &precond_ptr) -> int {
     *     return LinearSolver::solve<VectorType>(*operator,
     *                                            solution,
     *                                            rhs,
     *                                            data.linear_solver,
     *                                            *precond_ptr,
     *                                            "operation");
     *   },
     * preconditioner_used);
     */
    PreconditionerObjectType
    compute_preconditioner();

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
