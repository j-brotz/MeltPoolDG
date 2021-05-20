/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

using namespace dealii;

namespace MeltPoolDG
{
  namespace NormalVector
  {
    template <int dim, typename number = double>
    class NormalVectorOperator
      : public OperatorBase<dim,
                            number,
                            LinearAlgebra::distributed::BlockVector<number>,
                            LinearAlgebra::distributed::Vector<number>>
    {
    public:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using VectorizedArrayType = VectorizedArray<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;

      NormalVectorOperator(const ScratchData<dim> &scratch_data_in,
                           const double            damping_in,
                           const unsigned int      normal_dof_idx_in,
                           const unsigned int      normal_quad_idx_in,
                           const unsigned int      ls_dof_idx_in);

      void
      assemble_matrixbased(const VectorType &level_set_in,
                           SparseMatrixType &matrix,
                           BlockVectorType & rhs) const override;

      /*
       *  matrix-free utility
       */

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const override;

      void
      create_rhs(BlockVectorType &dst, const VectorType &src) const override;

      static void
      get_unit_normals_at_quadrature(const FEValues<dim> &        fe_values,
                                     const BlockVectorType &      normal_vector_field_in,
                                     std::vector<Tensor<1, dim>> &unit_normal_at_quadrature,
                                     const double                 zero = 1e-16);

    private:
      const ScratchData<dim> &scratch_data;

      double             damping;
      const unsigned int normal_dof_idx;
      const unsigned int normal_quad_idx;
      const unsigned int ls_dof_idx;
    };
  } // namespace NormalVector
} // namespace MeltPoolDG
