#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  class HeatCutOperator : public OperatorBase<dim, number>
  {
    // to avoid compiler warnings regarding hidden overridden functions
    using OperatorBase<dim, number>::vmult;
    using OperatorBase<dim, number>::assemble_matrixbased;
    using OperatorBase<dim, number>::create_rhs;
    using OperatorBase<dim, number>::compute_inverse_diagonal_from_matrixfree;

  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

    const ScratchData<dim>                     &scratch_data;
    const HeatData<number>                     &heat_data;
    const MaterialData<number>                 &material_data;
    const Evaporation::EvaporationData<number> &evapor_data;
    const unsigned int                          temp_dof_idx;
    const unsigned int                          temp_hanging_nodes_dof_idx;
    const unsigned int                          temp_quad_idx;

    const VectorType &temperature;

    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
      &mapping_info_surface;
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
      &mapping_info_cells;

    // use FE_DGQ for FEPointEvaluation (DoF numbering reasons)
    const dealii::FE_DGQ<dim>   fe_point_temp;
    const unsigned int          n_dofs_per_cell;
    const dealii::FESystem<dim> fe_point_vel;
    const unsigned int          n_dofs_per_cell_vel;

    // coefficients for weighted average operator for two-phase case
    const number kappa_l;
    const number kappa_g;

    const EvaluationFlags::EvaluationFlags evaluation_flags_surface;
    static constexpr unsigned int          n_lanes = dealii::VectorizedArray<number>::size();

    // weighted averaged Nitsche term factor for two-phase case
    number weighted_nitsche_factor;

    number cell_side_length = 0.;
    number ost_factor_implicit; // delta_t * theta
    number ost_factor_explicit; // delta_t * (1. - theta)
    number nitsche_factor;      // delta_t * gamma_Gamma / h

    // TODO boundary conditions, radiation etc.

    // TODO evaporation model

    // optional: laser intensity and direction
    std::shared_ptr<const dealii::Function<dim, number>> laser_intensity_profile;
    dealii::Tensor<1, dim, number>                       laser_direction;

    // optional: flow velocity for internal convection
    const unsigned int vel_dof_idx;
    const VectorType  *velocity;

    // optional: melting/solidification effects
    const bool do_solidification;

  public:
    HeatCutOperator(const ScratchData<dim>                     &scratch_data_in,
                    const HeatData<number>                     &heat_data_in,
                    const MaterialData<number>                 &material_data_in,
                    const Evaporation::EvaporationData<number> &evapor_data_in,
                    const unsigned int                          temp_dof_idx_in,
                    const unsigned int                          temp_hanging_nodes_dof_idx_in,
                    const unsigned int                          temp_quad_idx_in,
                    const VectorType                           &temperature_in,
                    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
                      &mapping_info_surface_in,
                    const std::vector<std::shared_ptr<
                      dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
                                      &mapping_info_cells_in,
                    const bool         do_solidification_in,
                    const unsigned int vel_dof_idx_in = 0,
                    const VectorType  *velocity_in    = nullptr);

    void
    register_laser_intensity_function_and_direction(
      std::shared_ptr<const dealii::Function<dim, number>> laser_intensity_profile_in,
      const dealii::Tensor<1, dim, number>                &laser_direction_in);

    void
    reinit() final;

    void
    init_time_advance(const double dt);

    void
    update_ghost_values() const;

    /*
     *    matrix-free implementation
     */
    void
    vmult(VectorType &dst, const VectorType &src) const final;

    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const final;

    /**
     * compute residual (-R)
     */
    void
    create_rhs(VectorType &residual, const VectorType &temperature_old) const final;

    /**
     * computes the L2 norm of the @param solution on the cut domain.
     */
    number
    compute_cut_L2_norm(const VectorType &solution) const;

  private:
    /*
     * Local appliers for consitent tangent modulus
     */
    void
    local_apply_domain_tangent(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &dst,
      const VectorType                                                       &src,
      const std::pair<unsigned int, unsigned int>                            &cell_range) const;

    void
    local_apply_inner_face_tangent(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &dst,
      const VectorType                                                       &src,
      const std::pair<unsigned int, unsigned int>                            &face_range) const;

    void
    local_apply_boundary_face_tangent(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &dst,
      const VectorType                                                       &src,
      const std::pair<unsigned int, unsigned int>                            &face_range) const;

    /*
     * Local appliers for residual
     */
    void
    local_apply_domain_residual(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
      VectorType                                                             &residual,
      const VectorType                                                       &temperature_new,
      const std::pair<unsigned int, unsigned int>                            &cell_range) const;

    void
    local_apply_inner_face_residual(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &residual,
      const VectorType                                                       &temperature_new,
      const std::pair<unsigned int, unsigned int>                            &face_range) const;

    void
    local_apply_boundary_face_residual(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &residual,
      const VectorType                                                       &temperature_new,
      const std::pair<unsigned int, unsigned int>                            &face_range) const;

    /**
     * The setup for dealii::MatrixFreeTools::internal::compute_diagonal and
     * dealii::MatrixFreeTools::internal::compute_matrix is identical. To avoid duplicate code this
     * internal function can handle both operations. Choose which operation to perform using
     * @param do_diagonal: `true` for compute_diagonal and `false` for compute_matrix.
     */
    void
    internal_compute_diagonal_or_system_matrix(
      [[maybe_unused]] VectorType                             &diagonal,
      [[maybe_unused]] dealii::TrilinosWrappers::SparseMatrix &system_matrix,
      const bool                                               do_diagonal) const;
  };
} // namespace MeltPoolDG::Heat
