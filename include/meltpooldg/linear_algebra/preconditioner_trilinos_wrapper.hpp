/**
 * @brief Wrapper class for deal.II matrix-based preconditioners.
 */

#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/core/scratch_data.hpp>

#include <any>
#include <type_traits>

namespace MeltPoolDG
{
  /**
   * A concept for operators that are compatible with the wrapper for deal.II matrix based
   * preconditioners when no external system matrix can be provided when the preconditioner is
   * updated.
   */
  template <typename OperatorType, typename VectorType>
  concept DealiipreconditionerWrapperOperatorType =
    requires(const OperatorType op, dealii::TrilinosWrappers::SparseMatrix &matrix) {
      /**
       * Compute the matrix representation of the system matrix and store it in the given matrix.
       */
      op.compute_system_matrix_from_matrixfree(matrix);
    };

  /**
   * Wrapper class for deal.II matrix-based preconditioners.
   */
  template <int dim, typename VectorType, typename DealiiPreconditionerType, typename OperatorType>
  class DealiiPreconditionerWrapper
  {
  public:
    /**
     * Constructor.
     *
     * @param operator_in Operator object used to compute the system matrix representation (when
     * computing matrix-free).
     * @param do_matrix_free Flag indicating whether the operator is used for matrix-free
     * computations.
     */
    explicit DealiiPreconditionerWrapper(const OperatorType *operator_in,
                                         const bool          do_matrix_free = true)
      : eq_operator(operator_in)
      , do_matrix_free(do_matrix_free)
    {}

    /**
     * Apply the preconditioner to the given @p src vector and store the result in the @p dst vector.
     *
     * @param dst Vector in which the result is stored.
     * @param src Source vector to which the preconditioner is applied.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      // deal.II trilinos preconditioner do not support vmult(BlockVectorType& BlockVectorType&).
      // This is done manually if required.
      // TODO: Can this be done more elegant?
      if constexpr (std::is_same_v<dealii::LinearAlgebra::distributed::BlockVector<
                                     typename VectorType::value_type>,
                                   VectorType>)
        for (unsigned int b = 0; b < dst.n_blocks(); ++b)
          preconditioner.vmult(dst.block(b), src.block(b));
      else
        preconditioner.vmult(dst, src);
    }

    /**
     * Update the preconditioner. For the matrix-free case this means to compute the matrix
     * representation and store it internally while for the matrix-base case it is required to pass
     * a pointer to the externally stored system matrix which is then used internally.
     *
     * @param external_preconditioner_matrix Pointer to the externally stored system matrix (only
     * required when computing matrix-base)
     *
     * @throws Exception if the no pointer is provided AND the operator does not support to
     * calculate a matrix-representaiton.
     */
    void
    update(const std::any &external_preconditioner_matrix = std::any())
    {
      dealii::TrilinosWrappers::SparseMatrix *preconditioner_matrix_ptr;
      if (not external_preconditioner_matrix.has_value())
        {
          if constexpr (DealiipreconditionerWrapperOperatorType<OperatorType, VectorType>)
            {
              Assert(eq_operator != nullptr,
                     dealii::ExcMessage("The provided operator has no valid address (nullptr)."));
              eq_operator->compute_system_matrix_from_matrixfree(preconditioner_system_matrix);
              preconditioner_matrix_ptr = &preconditioner_system_matrix;
            }
          else
            {
              AssertThrow(false,
                          dealii::ExcMessage(
                            "Neither a rule to compute the preconditioner matrix nor an external"
                            " preconditioner matrix are provided."));
            }
        }
      else
        {
          Assert(external_preconditioner_matrix.has_value(),
                 dealii::ExcMessage(
                   "No external system matrix has been passed to the preconditioner!"));
          preconditioner_matrix_ptr =
            std::any_cast<dealii::TrilinosWrappers::SparseMatrix *>(external_preconditioner_matrix);
        }
      preconditioner.initialize(*preconditioner_matrix_ptr,
                                typename DealiiPreconditionerType::AdditionalData());
    }

    /**
     * Initialize the internal data structures. In the matrix-based case this function does
     * effectively do nothing. However, in the matrix-free case memory is allocated for the (sparse)
     * preconditioner matrix.
     *
     * @param scratch_data Scratch data object to get relevant dof information for setting up the
     * sparse matrix.
     * @param dof_idx Relevant dof index in the scratch data object.
     */
    void
    reinit(const ScratchData<dim> &scratch_data, const unsigned int dof_idx)
    {
      if (eq_operator != nullptr && do_matrix_free)
        {
          dsp.reinit(scratch_data.get_dof_handler(dof_idx).n_dofs(),
                     scratch_data.get_dof_handler(dof_idx).n_dofs(),
                     scratch_data.get_locally_relevant_dofs(dof_idx));
          if (scratch_data.is_FE_DGQ(dof_idx))
            {
              dealii::DoFTools::make_flux_sparsity_pattern(scratch_data.get_dof_handler(dof_idx),
                                                           dsp,
                                                           scratch_data.get_constraint(dof_idx),
                                                           false);
            }
          else
            {
              dealii::DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(dof_idx),
                                                      dsp,
                                                      scratch_data.get_constraint(dof_idx));
            }
          dealii::SparsityTools::distribute_sparsity_pattern(
            dsp,
            scratch_data.get_locally_owned_dofs(dof_idx),
            scratch_data.get_mpi_comm(),
            scratch_data.get_locally_relevant_dofs(dof_idx));

          preconditioner_system_matrix.reinit(scratch_data.get_locally_owned_dofs(dof_idx),
                                              scratch_data.get_locally_owned_dofs(dof_idx),
                                              dsp,
                                              scratch_data.get_mpi_comm());
        }
    }

  private:
    DealiiPreconditionerType               preconditioner;
    dealii::TrilinosWrappers::SparseMatrix preconditioner_system_matrix;
    dealii::DynamicSparsityPattern         dsp;

    const OperatorType *eq_operator;
    const bool          do_matrix_free;
  };

  /**
   * Identity preconditoner. The deal.II identity preconditioner is not used here in order to match
   * the MeltPoolDG preconditioner interface.
   */
  template <int dim, typename VectorType>
  class IdentityPreconditioner
  {
  public:
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = src;
    }

    void
    update(const std::any &) const
    {}

    void
    reinit(const ScratchData<dim> &, unsigned int) const
    {}
  };
} // namespace MeltPoolDG