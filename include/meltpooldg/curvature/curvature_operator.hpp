/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>

// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>

using namespace dealii;

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
  template <int dim, typename number = double>
  class CurvatureOperator : public OperatorBase<dim, number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using OperatorBase<dim, number>::vmult;
    using OperatorBase<dim, number>::assemble_matrixbased;
    using OperatorBase<dim, number>::create_rhs;
    using OperatorBase<dim, number>::compute_inverse_diagonal_from_matrixfree;

  private:
    using VectorType          = LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
    using VectorizedArrayType = VectorizedArray<number>;
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
    using vector              = Tensor<1, dim, VectorizedArray<number>>;
    using scalar              = VectorizedArray<number>;

    const ScratchData<dim> &scratch_data;

    const CurvatureData<double> &curvature_data;

    const unsigned int curv_dof_idx;
    const unsigned int curv_quad_idx;
    const unsigned int normal_dof_idx;
    const double       tolerance_normal_vector;

    const unsigned int ls_dof_idx;

    // optional parameters for narrow band
    const VectorType *solution_level_set;

    AlignedVector<VectorizedArray<double>> damping;

  public:
    CurvatureOperator(const ScratchData<dim>      &scratch_data_in,
                      const CurvatureData<double> &curvature_data,
                      const unsigned int           curv_dof_idx_in,
                      const unsigned int           curv_quad_idx_in,
                      const unsigned int           normal_dof_idx_in,
                      const unsigned int           ls_dof_idx_in,
                      const VectorType            *solution_level_set_in = nullptr);

    void
    assemble_matrixbased(const BlockVectorType &solution_normal_vector_in,
                         SparseMatrixType      &matrix,
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
      TrilinosWrappers::SparseMatrix &system_matrix) const final;

    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    void
    reinit() final;

  private:
    void
    tangent_local_cell_operation(FECellIntegrator<dim, 1, number> &curv_vals,
                                 FECellIntegrator<dim, 1, number> &level_set_vals,
                                 const bool                        do_reinit_cells) const;
  };
} // namespace MeltPoolDG::LevelSet
