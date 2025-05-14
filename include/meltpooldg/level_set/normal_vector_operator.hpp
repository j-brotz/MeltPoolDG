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

    /*
     * Additional functions for wetting boundary conditions (inhomogenous Dirichlet boundary
     * conditions).
     */

    /**
     * @brief Set local DoF indices of wetting boundary condition in member variable
     *
     * @param[in] p_wetting_bc_local_indices Local indices DoF where the wetting boundary condition
     * is applied.
     */
    void
    set_wetting_bc_indices(const std::vector<unsigned int> &p_wetting_bc_local_indices);

    /**
     * @brief Set local DoF indices of contact angle boundary condition in member variable
     *
     * @param[in] p_contact_angle_bc_local_indices Local indices DoF where the contact angle
     * boundary condition is applied.
     */
    void
    set_contact_angle_bc_indices(const std::vector<unsigned int> &p_contact_angle_bc_local_indices);

    /**
     * @brief Disable pre-treatment and post-treatment respectively before and after the
     * NormalVectorOperator::vmult operation.
     * This is used to constrain DoFs according to specified wetting boundary conditions.
     */
    void
    enable_pre_post();

    /**
     * @brief Disable pre-treatment and post-treatment respectively before and after the
     * NormalVectorOperator::vmult operation.
     */
    void
    disable_pre_post();

    /**
     * @brief Zero out values of the source vector that corresponds to wetting boundary conditions
     * and store values in @p wetting_constraints_values_temp.
     *
     * @param[out] dst Destination vector
     *
     * @param[in] src_in Input vector
     *
     * @remark Here, @p dst is unused.
     */
    void
    do_pre_vmult([[maybe_unused]] BlockVectorType &dst, const BlockVectorType &src_in) const;

    /**
     * @brief Copy values previously stored in @p wetting_constraints_values_temp with
     * NormalVectorOperator::do_pre_vmult into @p dst. This ensures that correct boundary condition
     * values are applied.
     *
     * @param[out] dst Destination vector
     *
     * @param[in] src_in Input vector
     *
     * @remark Here, @p src_in is unused.
     */
    void
    do_post_vmult(BlockVectorType &dst, [[maybe_unused]] const BlockVectorType &src_in) const;

    /**
     * @brief Zero out rows of the matrix corresponding to wetting boundary conditions and set the
     * diagonal to 1.0. This translates into Dirichlet boundary conditions.
     *
     * @param[out] system_matrix Matrix of the linear system solved
     */
    void
    post_system_matrix_compute(dealii::TrilinosWrappers::SparseMatrix &system_matrix) const;

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
    /** Boolean indicating if a pre-treatment and a post-treatment respectively before and after the
     * NormalVectorOperator::vmult operation. */
    mutable bool do_pre_post = false;
    /// Local DoF indices corresponding to wetting boundary conditions
    std::vector<unsigned int> wetting_bc_local_indices;
    /// Temporary storage of boundary condition values
    mutable std::vector<std::vector<number>> wetting_constraints_values_temp;
    /// Local DoF indices corresponding to contact angle boundary conditions
    std::vector<unsigned int> contact_angle_bc_local_indices;
    /// Temporary storage of boundary condition values
    mutable std::vector<std::vector<number>> contact_angle_constraints_values_temp;
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
