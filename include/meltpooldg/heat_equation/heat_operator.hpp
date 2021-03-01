/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>

namespace MeltPoolDG::HeatEquation
{
  using namespace dealii;

  template <int dim, typename number = double>
  class HeatOperator : public OperatorBase<number,
                                           LinearAlgebra::distributed::Vector<number>,
                                           LinearAlgebra::distributed::Vector<number>>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
    using VectorizedArrayType = VectorizedArray<number>;

    const ScratchData<dim> &scratch_data;
    const HeatData<number> &data;
    const unsigned int      temp_dof_idx;
    const unsigned int      temp_quad_idx;

  public:
    HeatOperator(const ScratchData<dim> &scratch_data_in,
                 const HeatData<number> &data_in,
                 const unsigned int      temp_dof_idx_in,
                 const unsigned int      temp_quad_idx_in)
      // clang-format off
    : scratch_data        ( scratch_data_in       )
    , data                ( data_in               )
    , temp_dof_idx        ( temp_dof_idx_in       )
    , temp_quad_idx       ( temp_quad_idx_in       )
    {
      this->reset_indices(temp_dof_idx_in, temp_quad_idx_in);
    }
    // clang-format on

    /*
     *    this is the matrix-based implementation of the rhs and the matrix
     *    @todo: this could be improved by using the WorkStream functionality of dealii
     */

    void
    assemble_matrixbased([[maybe_unused]] const VectorType &advected_field_old,
                         [[maybe_unused]] SparseMatrixType &matrix,
                         [[maybe_unused]] VectorType &      rhs) const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    /*
     *    matrix-free implementation
     */
    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    create_rhs(VectorType &dst, const VectorType &src) const override
    {
      AssertThrow(false, ExcNotImplemented());
    }
  };
} // namespace MeltPoolDG::HeatEquation
