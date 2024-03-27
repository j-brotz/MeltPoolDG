/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// MeltPoolDG
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::LevelSet
{
  static std::map<std::string, double> get_generalized_theta = {
    {"explicit_euler", 0.0},
    {"implicit_euler", 1.0},
    {"crank_nicolson", 0.5},
  };

  using namespace dealii;

  template <int dim, typename number = double>
  class AdvectionDiffusionOperator : public OperatorBase<dim, number>
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
    AdvectionDiffusionOperator(const ScratchData<dim>               &scratch_data_in,
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
    assemble_matrixbased(const VectorType &advected_field_old,
                         SparseMatrixType &matrix,
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
      TrilinosWrappers::SparseMatrix &system_matrix) const final;

    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    void
    reinit() final;

    void
    prepare() final;

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
    post_system_matrix_compute(TrilinosWrappers::SparseMatrix &system_matrix) const;


  private:
    void
    tangent_local_cell_operation(FECellIntegrator<dim, 1, number>   &advected_field,
                                 FECellIntegrator<dim, dim, number> &velocity_vals,
                                 const bool                          do_reinit_cells) const;

    void
    compute_stabilization_parameter(const FECellIntegrator<dim, dim, number> &velocity_vals) const;

    const ScratchData<dim>                        &scratch_data;
    const VectorType                              &advection_velocity;
    const AdvectionDiffusionData<number>          &data;
    const unsigned int                             velocity_dof_idx;
    const unsigned int                             advec_diff_quad_idx;
    double                                         theta;
    mutable AlignedVector<VectorizedArray<double>> stab_param;
    mutable bool                                   do_update_stab_param = true;

    mutable bool do_pre_post = false;

    std::vector<unsigned int>   inflow_outflow_bc_local_indices;
    mutable std::vector<double> inflow_outflow_constraints_values_temp;
  };
} // namespace MeltPoolDG::LevelSet
