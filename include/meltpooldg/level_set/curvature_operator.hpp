#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/curvature_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

namespace MeltPoolDG::LevelSet
{
  /**
   *  This function calculates the curvature of the current level set function being
   *  the solution of an intermediate projection step
   *
   *              (w, κ)   +   η_κ (∇w, ∇κ)  = (w,∇·n_ϕ)
   *                    Ω                  Ω            Ω
   *
   *  with test function w, curvature κ, damping parameter η_κ and the normal to the
   *  level set function n_ϕ.
   *
   */
  template <int dim, typename number>
  class CurvatureOperator : public OperatorMatrixFree<dim, number>,
                            public OperatorMatrixBased<dim, number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using OperatorMatrixBased<dim, number>::compute_system_matrix_and_rhs;
    using OperatorMatrixFree<dim, number>::vmult;
    using OperatorMatrixFree<dim, number>::create_rhs;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;

  private:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using SparseMatrixType    = dealii::TrilinosWrappers::SparseMatrix;
    using vector              = dealii::Tensor<1, dim, dealii::VectorizedArray<number>>;
    using scalar              = dealii::VectorizedArray<number>;

    const ScratchData<dim, dim, number> &scratch_data;

    const CurvatureData<number> &curvature_data;

    const unsigned int curv_dof_idx;
    const unsigned int curv_quad_idx;
    const unsigned int normal_dof_idx;
    const number       tolerance_normal_vector;

    const unsigned int ls_dof_idx;

    // optional parameters for narrow band
    const VectorType *solution_level_set;

    dealii::AlignedVector<dealii::VectorizedArray<number>> damping;

  public:
    CurvatureOperator(const ScratchData<dim, dim, number> &scratch_data_in,
                      const CurvatureData<number>         &curvature_data,
                      const unsigned int                   curv_dof_idx_in,
                      const unsigned int                   curv_quad_idx_in,
                      const unsigned int                   normal_dof_idx_in,
                      const unsigned int                   ls_dof_idx_in,
                      const VectorType                    *solution_level_set_in = nullptr);

    void
    compute_system_matrix_and_rhs(const BlockVectorType &solution_normal_vector_in,
                                  VectorType            &rhs) const final;

    /*
     *  matrix-free utility
     */

    void
    vmult(VectorType &dst, const VectorType &src) const final;

    void
    create_rhs(VectorType &dst, const BlockVectorType &src) const final;

    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const final;

    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    void
    reinit() final;

  private:
    void
    tangent_local_cell_operation(dealii::FECellIntegrator<dim, 1, number> &curv_vals,
                                 dealii::FECellIntegrator<dim, 1, number> &level_set_vals,
                                 const bool                                do_reinit_cells) const;
  };
} // namespace MeltPoolDG::LevelSet
