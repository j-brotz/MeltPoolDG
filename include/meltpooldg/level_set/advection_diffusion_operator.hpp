#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/advection_diffusion_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <map>
#include <string>
#include <vector>


namespace MeltPoolDG::LevelSet
{
  static std::map<std::string, double> get_generalized_theta = {
    {"explicit_euler", 0.0},
    {"implicit_euler", 1.0},
    {"crank_nicolson", 0.5},
  };

  template <int dim, typename number>
  class AdvectionDiffusionOperator : public OperatorMatrixFree<dim, number>,
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
    using SparseMatrixType    = dealii::TrilinosWrappers::SparseMatrix;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using vector              = dealii::Tensor<1, dim, dealii::VectorizedArray<number>>;
    using scalar              = dealii::VectorizedArray<number>;

  public:
    AdvectionDiffusionOperator(const ScratchData<dim, dim, number>  &scratch_data_in,
                               const VectorType                     &advection_velocity_in,
                               const AdvectionDiffusionData<number> &data_in,
                               unsigned int                          dof_idx_in,
                               unsigned int                          quad_idx_in,
                               unsigned int                          velocity_dof_idx_in);

    /*
     *    this is the matrix-based implementation of the rhs and the matrix
     *    @todo: this could be improved by using the WorkStream functionality of deal.II
     */

    void
    compute_system_matrix_and_rhs(const VectorType &advected_field_old,
                                  VectorType       &rhs) const final;
    /*
     *    matrix-free implementation
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

    void
    pre() final;

    /**
     * additional functions for inflow/outflow boundary conditions
     */
    void
    set_inflow_outflow_bc(std::vector<unsigned int> inflow_outflow_bc_local_indices_);

    void
    enable_pre_post();

    void
    disable_pre_post();

    void
    do_pre_vmult([[maybe_unused]] VectorType &dst, const VectorType &src_in) const;

    void
    do_post_vmult(VectorType &dst, [[maybe_unused]] const VectorType &src_in) const;

    void
    post_system_matrix_compute(dealii::TrilinosWrappers::SparseMatrix &system_matrix) const;


  private:
    void
    tangent_local_cell_operation(FECellIntegrator<dim, 1, number>   &advected_field,
                                 FECellIntegrator<dim, dim, number> &velocity_vals,
                                 const bool                          do_reinit_cells) const;

    void
    compute_stabilization_parameter(const FECellIntegrator<dim, dim, number> &velocity_vals) const;

    const ScratchData<dim, dim, number>                           &scratch_data;
    const VectorType                                              &advection_velocity;
    const AdvectionDiffusionData<number>                          &data;
    const unsigned int                                             velocity_dof_idx;
    const unsigned int                                             advec_diff_quad_idx;
    number                                                         theta;
    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> stab_param;
    mutable bool                                                   do_update_stab_param = true;

    mutable bool do_pre_post = false;

    std::vector<unsigned int>   inflow_outflow_bc_local_indices;
    mutable std::vector<number> inflow_outflow_constraints_values_temp;
  };
} // namespace MeltPoolDG::LevelSet
