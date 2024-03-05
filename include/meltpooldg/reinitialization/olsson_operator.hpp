/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/reinitialization/reinitialization_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    template <int dim, typename number = double>
    class OlssonOperator : public OperatorBase<dim, number>
    {
      //@todo: to avoid compiler warnings regarding hidden overriden functions
      using OperatorBase<dim, number>::vmult;
      using OperatorBase<dim, number>::assemble_matrixbased;
      using OperatorBase<dim, number>::create_rhs;
      using OperatorBase<dim, number>::compute_inverse_diagonal_from_matrixfree;

    private:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
      using VectorizedArrayType = VectorizedArray<number>;
      using vector              = Tensor<1, dim, VectorizedArray<number>>;
      using scalar              = VectorizedArray<number>;

    public:
      OlssonOperator(const ScratchData<dim>             &scratch_data_in,
                     const ReinitializationData<number> &reinit_data_in,
                     const int                           ls_n_subdivisions,
                     const BlockVectorType              &n_in,
                     const unsigned int                  reinit_dof_idx_in,
                     const unsigned int                  reinit_quad_idx_in,
                     const unsigned int                  ls_dof_idx_in,
                     const unsigned int                  normal_dof_idx_in);

      /*
       *    this is the matrix-based implementation of the rhs and the system_matrix
       *    @todo: this could be improved by using the WorkStream functionality of dealii
       */

      void
      assemble_matrixbased(const VectorType &levelset_old,
                           SparseMatrixType &matrix,
                           VectorType       &rhs) const final;

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
        TrilinosWrappers::SparseMatrix &system_matrix) const final;

      void
      compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

      void
      reinit() final;

    private:
      void
      tangent_local_cell_operation(FECellIntegrator<dim, 1, number> &delta_psi) const;

    private:
      const ScratchData<dim>             &scratch_data;
      const ReinitializationData<number> &reinit_data;
      const double                        ls_n_subdivisions;
      const BlockVectorType              &normal_vec;
      const unsigned int                  reinit_quad_idx;
      const unsigned int                  normal_dof_idx;
      const unsigned int                  ls_dof_idx;

      const double tolerance_normal_vector;

      AlignedVector<VectorizedArray<double>>                         diffusion_length;
      mutable AlignedVector<Tensor<1, dim, VectorizedArray<double>>> unit_normal;
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
