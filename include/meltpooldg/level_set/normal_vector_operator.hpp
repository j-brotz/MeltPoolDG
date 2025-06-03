#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/level_set/normal_vector_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class NormalVectorOperator : public OperatorMatrixFree<dim, number>,
                               public OperatorMatrixBased<dim, number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using OperatorMatrixBased<dim, number>::compute_system_matrix_and_rhs;
    using OperatorMatrixFree<dim, number>::vmult;
    using OperatorMatrixFree<dim, number>::create_rhs;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;

  public:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using SparseMatrixType    = dealii::TrilinosWrappers::SparseMatrix;

    NormalVectorOperator(const ScratchData<dim, dim, number> &scratch_data_in,
                         const NormalVectorData<number>      &normal_vector_data_in,
                         const std::array<unsigned int, dim> &normal_dof_indices_per_block_in,
                         const unsigned int                   normal_quad_idx_in,
                         const unsigned int                   ls_dof_idx_in,
                         const VectorType                    *solution_level_set_in = nullptr);

    void
    compute_system_matrix_and_rhs(const VectorType &level_set_in, BlockVectorType &rhs) const final;

    /*
     *  matrix-free utility
     */

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const final;

    void
    create_rhs(BlockVectorType &dst, const VectorType &src) const final;

    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const final;

    void
    compute_inverse_diagonal_from_matrixfree(BlockVectorType &diagonal) const final;

    static void
    get_unit_normals_at_quadrature(
      const dealii::FEValues<dim>                 &fe_values,
      const BlockVectorType                       &normal_vector_field_in,
      std::vector<dealii::Tensor<1, dim, number>> &unit_normal_at_quadrature,
      const number                                 zero = 1e-16);
    void
    reinit() final;

  private:
    void
    tangent_local_cell_operation(FECellIntegrator<dim, 1, number> &normal_vals,
                                 FECellIntegrator<dim, 1, number> &level_set_vals,
                                 const bool                        do_reinit_cells) const;

  private:
    const ScratchData<dim, dim, number> &scratch_data;
    const NormalVectorData<number>      &normal_vector_data;

    const std::array<unsigned int, dim> normal_dof_indices_per_block;
    const unsigned int                  normal_quad_idx;
    const unsigned int                  ls_dof_idx;

    // optional parameters for narrow band
    const VectorType *solution_level_set;

    dealii::AlignedVector<dealii::VectorizedArray<number>> damping;

    /*
     * For boundary conditions with wetting
     */
  };

  /**
   * Matrix-free
   *
   * For a given @param cell, compute the cell_size dependent filter parameter
   *
   *    scale_factor * h^2
   *
   * using a given @param scale_factor.
   *
   * @todo: move to normal_vector_utils.hpp
   */
  template <int dim, typename number>
  inline dealii::VectorizedArray<number>
  compute_cell_size_dependent_filter_parameter_mf(const ScratchData<dim, dim, number> &scratch_data,
                                                  const unsigned int                   dof_idx,
                                                  const unsigned int                   cell_idx,
                                                  const number                         scale_factor)
  {
    const number n_subdivisions =
      scratch_data.is_FE_Q_iso_Q_1(dof_idx) ? scratch_data.get_degree(dof_idx) : 1;
    return dealii::Utilities::fixed_power<2>(
             std::max(dealii::VectorizedArray<number>(scratch_data.get_min_cell_size()),
                      scratch_data.get_cell_sizes()[cell_idx] / (number)n_subdivisions)) *
           scale_factor;
  }

  /**
   * For a given @param cell, compute the cell_size dependent filter parameter
   *
   *    scale_factor * h^2
   *
   * using a given @param scale_factor.
   *
   * @todo: move to normal_vector_utils.hpp
   */
  template <int dim, typename number, typename cell_type>
  number
  compute_cell_size_dependent_filter_parameter(const ScratchData<dim, dim, number> &scratch_data,
                                               const unsigned int                   dof_idx,
                                               const cell_type                      cell,
                                               const number                         scale_factor)
  {
    const number n_subdivisions =
      scratch_data.is_FE_Q_iso_Q_1(dof_idx) ? scratch_data.get_degree(dof_idx) : 1;

    return dealii::Utilities::fixed_power<2>(
             std::max(scratch_data.get_min_cell_size(),
                      cell->diameter() / (std::sqrt(dim) * n_subdivisions))) *
           scale_factor;
  }
} // namespace MeltPoolDG::LevelSet
