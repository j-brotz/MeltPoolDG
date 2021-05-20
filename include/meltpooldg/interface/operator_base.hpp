#pragma once
// dealii
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
// MeltPoolDG
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim,
            typename number           = double,
            typename DoFVectorType    = LinearAlgebra::distributed::Vector<number>,
            typename SrcRhsVectorType = DoFVectorType>
  class OperatorBase
  {
  private:
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;

  public:
    virtual ~OperatorBase() = default;

    virtual void
    assemble_matrixbased(const SrcRhsVectorType &src,
                         SparseMatrixType &      matrix,
                         DoFVectorType &         rhs) const
    {
      (void)src;
      (void)matrix;
      (void)rhs;
      AssertThrow(false,
                  ExcMessage("assemble_matrixbased for the requested operator not implemented"));
    }

    virtual void
    create_rhs(DoFVectorType &dst, const SrcRhsVectorType &src) const
    {
      (void)dst;
      (void)src;
      AssertThrow(false, ExcMessage("create_rhs for the requested operator not implemented"));
    }

    virtual void
    vmult(DoFVectorType &dst, const DoFVectorType &src) const
    {
      (void)dst;
      (void)src;
      AssertThrow(false, ExcMessage("vmult for the requested operator not implemented"));
    }

    virtual void
    print_me() const
    {}

    void
    set_time_increment(const double dt);

    void
    reset_indices(const unsigned int dof_idx_in, const unsigned int quad_idx_in);

    void
    initialize_matrix_based(const ScratchData<dim> &scratch_data);

    /**
     * Compute the modified right-handside for (inhomogeneous) dirichlet boundary conditions G
     *
     * A * x = B
     *
     * We actually solve
     *
     * A * x_0 = b - A * G
     *
     * with zero Dirichlet boundary conditions.
     */
    void
    create_rhs_and_apply_dirichlet_mf(DoFVectorType &         rhs,
                                      const SrcRhsVectorType &src,
                                      const ScratchData<dim> &scratch_data,
                                      const unsigned int      dof_idx,
                                      const unsigned int      dof_no_bc_idx);

    const SparseMatrixType &
    get_system_matrix() const;

    double              d_tau     = 0.0;
    double              d_tau_inv = 0.0;
    SparseMatrixType    system_matrix;
    SparsityPatternType dsp;

  protected:
    /*
     * dof_idx/quad_idx can be overwritten from the derived operator class by calling the
     * reset_indices function
     * */
    unsigned int dof_idx  = 0;
    unsigned int quad_idx = 0;
  };
} // namespace MeltPoolDG
