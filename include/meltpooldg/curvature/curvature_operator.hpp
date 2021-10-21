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

using namespace dealii;

namespace MeltPoolDG::Curvature
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
  class CurvatureOperator : public OperatorBase<dim,
                                                number,
                                                LinearAlgebra::distributed::Vector<number>,
                                                LinearAlgebra::distributed::BlockVector<number>>
  {
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

    // optional parameters for narrow band
    const bool         do_narrow_band;
    const unsigned int ls_dof_idx;
    const VectorType * solution_level_set;
    const double       narrow_band_threshold = 0.9999999;

  public:
    CurvatureOperator(const ScratchData<dim> &     scratch_data_in,
                      const CurvatureData<double> &curvature_data,
                      const unsigned int           curv_dof_idx_in,
                      const unsigned int           curv_quad_idx_in,
                      const unsigned int           normal_dof_idx_in,
                      const unsigned int           ls_dof_idx_in,
                      const bool                   do_narrow_band        = false,
                      const VectorType *           solution_level_set_in = nullptr);

    void
    assemble_matrixbased(const BlockVectorType &solution_normal_vector_in,
                         SparseMatrixType &     matrix,
                         VectorType &           rhs) const override;

    /*
     *  matrix-free utility
     */

    void
    vmult(VectorType &dst, const VectorType &src) const override;

    void
    create_rhs(VectorType &dst, const BlockVectorType &src) const override;
  };
} // namespace MeltPoolDG::Curvature
