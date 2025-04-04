/**
 * @brief Jacobi preconditioner for matrix-free operators.
 */

#pragma once

#include <meltpooldg/core/scratch_data.hpp>

#include <any>

namespace MeltPoolDG
{
  /**
   * A concept for matrix free operators that are compatible with the Jacobi preconditioner.
   */
  template <typename OperatorType, typename VectorType>
  concept JacobiPreconditionerOperatorType = requires(const OperatorType op, VectorType &diag) {
    /**
     * Compute the inverse of the diagonal of the system matrix and store it in the given vector.
     */
    op.compute_inverse_diagonal_from_matrixfree(diag);
  };


  template <int dim,
            typename number,
            typename VectorType,
            JacobiPreconditionerOperatorType<VectorType> OperatorType>
  class JacobiPreconditioner
  {
  public:
    /**
     * Constructor.
     *
     * @param operator_in Operator object used to compute the system matrix representation.
     */
    explicit JacobiPreconditioner(const OperatorType &operator_in)
      : eq_operator(operator_in)
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
      dst = src;
      dst.scale(inverse_diag);
    }

    /**
     * Update the preconditioner. This means to compute the inverse diagonal of the system matrix
     * and store it internally.
     *
     * @note The function takes a single value only to confirm to the interface. It is not used in
     * the fucniton.
     */
    void
    update(const std::any & = std::any())
    {
      eq_operator.compute_inverse_diagonal_from_matrixfree(inverse_diag);
    }

    /**
     * Initialize the internal data structures.
     *
     * @param scratch_data Scratch data object to get partitioning of vectors.
     * @param dof_idx Relevant dof index in the scratch data object.
     */
    void
    reinit(const ScratchData<dim, dim, number> &scratch_data, const unsigned int dof_idx)
    {
      scratch_data.initialize_dof_vector(inverse_diag, dof_idx);
    }

  private:
    //! Inverse values of the system matrix diagonal.
    VectorType inverse_diag;

    const OperatorType &eq_operator;
  };
} // namespace MeltPoolDG