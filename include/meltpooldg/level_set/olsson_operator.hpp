#pragma once
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/normal_vector_operator.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class OlssonOperator : public OperatorMatrixBased<dim, number>,
                         public OperatorMatrixFree<dim, number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using OperatorMatrixBased<dim, number>::compute_system_matrix_and_rhs;
    using OperatorMatrixFree<dim, number>::vmult;
    using OperatorMatrixFree<dim, number>::create_rhs;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;

  private:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType    = dealii::TrilinosWrappers::SparseMatrix;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using vector              = dealii::Tensor<1, dim, dealii::VectorizedArray<number>>;
    using scalar              = dealii::VectorizedArray<number>;

  public:
    OlssonOperator(const ScratchData<dim, dim, number> &scratch_data_in,
                   const ReinitializationData<number>  &reinit_data_in,
                   const int                            ls_n_subdivisions,
                   const BlockVectorType               &n_in,
                   const unsigned int                   reinit_dof_idx_in,
                   const unsigned int                   reinit_quad_idx_in,
                   const unsigned int                   ls_dof_idx_in,
                   const unsigned int                   normal_dof_idx_in);

    /*
     *    this is the matrix-based implementation of the rhs and the system_matrix
     *    @todo: this could be improved by using the WorkStream functionality of dealii
     */

    void
    compute_system_matrix_and_rhs(const VectorType &levelset_old, VectorType &rhs) const final;

    /*
     *    matrix-free implementation
     *
     */

    void
    vmult(VectorType &dst, const VectorType &src) const final;

    void
    create_rhs(VectorType &dst, const VectorType &src) const final;

    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const final;

    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    void
    reinit() final;

  private:
    void
    tangent_local_cell_operation(dealii::FECellIntegrator<dim, 1, number> &delta_psi) const;

  private:
    const ScratchData<dim, dim, number> &scratch_data;
    const ReinitializationData<number>  &reinit_data;
    const number                         ls_n_subdivisions;
    const BlockVectorType               &normal_vec;
    const unsigned int                   reinit_quad_idx;
    const unsigned int                   normal_dof_idx;
    const unsigned int                   ls_dof_idx;

    const number tolerance_normal_vector;

    dealii::AlignedVector<dealii::VectorizedArray<number>> diffusion_length;
    mutable dealii::AlignedVector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
      unit_normal;
  };
} // namespace MeltPoolDG::LevelSet
