#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/numbers.hpp>


namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim, typename number = double>
  class OperatorBase
  {
  private:
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;
    using VectorType          = LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;

  public:
    virtual ~OperatorBase() = default;

    /* ---------------------------------------------------------------------
     *
     * Matrix-based
     *
     * ---------------------------------------------------------------------*/
    virtual void
    assemble_matrixbased(const VectorType &, SparseMatrixType &, VectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    assemble_matrixbased(const BlockVectorType &, SparseMatrixType &, VectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    assemble_matrixbased(const VectorType &, SparseMatrixType &, BlockVectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    initialize_matrix_based(const ScratchData<dim> &scratch_data);

    const SparseMatrixType &
    get_system_matrix() const
    {
      return this->system_matrix;
    }

    SparseMatrixType &
    get_system_matrix()
    {
      return this->system_matrix;
    }

    /* ---------------------------------------------------------------------
     *
     * Matrix-free
     *
     * ---------------------------------------------------------------------*/
    virtual void
    create_rhs(VectorType &, const VectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    create_rhs(BlockVectorType &, const VectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    create_rhs(VectorType &, const BlockVectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    vmult(VectorType &, const VectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    vmult(BlockVectorType &, const BlockVectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    compute_system_matrix_from_matrixfree(TrilinosWrappers::SparseMatrix &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    compute_inverse_diagonal_from_matrixfree(VectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    compute_inverse_diagonal_from_matrixfree(BlockVectorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    /* ---------------------------------------------------------------------
     *
     * General
     *
     * ---------------------------------------------------------------------*/

    virtual void
    reinit()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    prepare()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    inline void
    reset_dof_index(const unsigned int dof_idx_in)
    {
      this->dof_idx = dof_idx_in;
    }

    inline void
    reset_time_increment(const double dt)
    {
      time_increment     = dt;
      time_increment_inv = 1. / time_increment;
    }

  protected:
    /*
     * matrix-based system matrix and sparsity pattern
     */
    SparseMatrixType    system_matrix;
    SparsityPatternType dsp;
    /*
     * dof_idx/quad_idx can be overwritten from the derived operator class by calling the
     * reset_indices function
     * */
    unsigned int dof_idx = numbers::invalid_unsigned_int;
    /*
     * time increment
     */
    double time_increment = 0.0;
    /*
     * reciprocal value of the time increment
     */
    double time_increment_inv = 0.0;
  };
} // namespace MeltPoolDG
