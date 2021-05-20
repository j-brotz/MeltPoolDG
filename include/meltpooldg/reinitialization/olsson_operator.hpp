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
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  namespace Reinitialization
  {
    using namespace dealii;

    template <int dim, typename number = double>
    class OlssonOperator : public OperatorBase<dim,
                                               number,
                                               LinearAlgebra::distributed::Vector<number>,
                                               LinearAlgebra::distributed::Vector<number>>
    {
    private:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
      using VectorizedArrayType = VectorizedArray<number>;
      using vector              = Tensor<1, dim, VectorizedArray<number>>;
      using scalar              = VectorizedArray<number>;

    public:
      OlssonOperator(const ScratchData<dim> &scratch_data_in,
                     BlockVectorType &       n_in,
                     const double &          constant_epsilon,
                     const double &          eps_scale_factor,
                     const unsigned int      dof_idx_in,
                     const unsigned int      quad_idx_in);

      /*
       *    this is the matrix-based implementation of the rhs and the system_matrix
       *    @todo: this could be improved by using the WorkStream functionality of dealii
       */

      void
      assemble_matrixbased(const VectorType &levelset_old,
                           SparseMatrixType &matrix,
                           VectorType &      rhs) const override;

      /*
       *    matrix-free implementation
       *
       */

      void
      vmult(VectorType &dst, const VectorType &src) const override;

      void
      create_rhs(VectorType &dst, const VectorType &src) const override;

      void
      set_normal_vector_field(const BlockVectorType &normal_vector);

    private:
      const ScratchData<dim> &scratch_data;

      double           eps = -1.0;
      double           eps_scale_factor;
      BlockVectorType &normal_vec;
      const double     tolerance_normal_vector;
    };
  } // namespace Reinitialization
} // namespace MeltPoolDG
