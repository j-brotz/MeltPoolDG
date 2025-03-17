#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporative_cooling.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/material.hpp>
#include <meltpooldg/utilities/material_data.hpp>

#include <memory>
#include <utility>
#include <vector>


namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  class HeatCutOperator : public OperatorMatrixFree<dim, number>
  {
  private:
    // to avoid compiler warnings regarding hidden overridden functions
    using OperatorMatrixFree<dim, number>::vmult;
    using OperatorMatrixFree<dim, number>::create_rhs;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;

    template <int n_components = 1>
    using DomainEval = dealii::FECellIntegrator<dim, n_components, number>;
    template <int n_components = 1>
    using PointEval =
      dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>>;
    using FaceEval = FEFaceIntegrator<dim, 1, number>;

    using VectorType       = typename OperatorMatrixFree<dim, number>::VectorType;
    using SparseMatrixType = typename OperatorMatrixFree<dim, number>::SparseMatrixType;

    const ScratchData<dim, dim, number> &scratch_data;
    const HeatData<number>              &heat_data;
    const Material<number>               material;

    // ScratchData's DoFHandler indices for ..
    const unsigned int temp_cut_dof_idx;        // .. CutFEM DoFs with Dirichlet BCs
    const unsigned int temp_cut_no_bc_dof_idx;  // .. CutFEM DoFs without Dirichlet BCs
    const unsigned int temp_cont_no_bc_dof_idx; // .. continuous DoFs without Dirichlet BCs
    // ScratchData's Quadrature index
    const unsigned int temp_quad_idx;

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

    static constexpr unsigned int n_lanes = dealii::VectorizedArray<number>::size();

    // weighted averaged Nitsche term factor for two-phase case
    number weighted_nitsche_factor;

    number cell_side_length = 0.;
    number ost_factor_implicit; // delta_t * theta
    number ost_factor_explicit; // delta_t * (1. - theta)
    number nitsche_factor;      // delta_t * gamma_Gamma / h

    // TODO boundary conditions, radiation etc.

    // optional: evaporative heat cooling
    std::unique_ptr<Evaporation::EvaporativeCooling<number>> evapor_cooling;

    // optional: laser intensity and direction
    std::shared_ptr<const dealii::Function<dim, number>> laser_intensity_profile;
    dealii::Tensor<1, dim, number>                       laser_direction;

    // optional: flow velocity for internal convection
    const unsigned int vel_dof_idx;
    const VectorType  *velocity;

    // optional: melting/solidification effects
    const bool do_solidification;

  public:
    HeatCutOperator(const ScratchData<dim, dim, number>        &scratch_data_in,
                    const HeatData<number>                     &heat_data_in,
                    const MaterialData<number>                 &material_data_in,
                    const Evaporation::EvaporationData<number> &evapor_data_in,
                    const unsigned int                          temp_cut_dof_idx_in,
                    const unsigned int                          temp_cut_no_bc_dof_idx_in,
                    const unsigned int                          temp_cont_no_bc_dof_idx_in,
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
    dealii::VectorizedArray<number>
    compute_qVapor(const dealii::VectorizedArray<number> &T) const;

    dealii::VectorizedArray<number>
    compute_qVapor_derivative(const dealii::VectorizedArray<number> &T) const;

    /*
     * Local appliers for consistent tangent modulus
     */
    void
    tangent_cell_operation_liquid(const unsigned int cell_index,
                                  DomainEval<>      &eval_l,
                                  DomainEval<>      *T_eval_l,
                                  DomainEval<dim>   *vel_eval,
                                  const bool         do_reinit_cell = true) const;

    void
    tangent_cell_operation_gas(const unsigned int cell_index,
                               DomainEval<>      &eval_g,
                               DomainEval<dim>   *vel_eval,
                               const bool         do_reinit_cell = true) const;

    void
    tangent_cell_operation_intersected_one_phase(const unsigned int cell_index,
                                                 DomainEval<>      &eval_cell_l,
                                                 PointEval<>       &eval_subdomain_l,
                                                 PointEval<>       *eval_interface_l,
                                                 DomainEval<>      *T_eval_cell_l,
                                                 PointEval<>       *T_eval_subdomain_l,
                                                 PointEval<>       *T_eval_interface_l,
                                                 DomainEval<dim>   *vel_eval,
                                                 PointEval<dim>    *vel_eval_subdomain_l,
                                                 const bool         do_reinit_cell = true) const;

    void
    tangent_cell_operation_intersected_two_phase(const unsigned int cell_index,
                                                 DomainEval<>      &eval_cell_l,
                                                 DomainEval<>      &eval_cell_g,
                                                 PointEval<>       &eval_subdomain_l,
                                                 PointEval<>       &eval_subdomain_g,
                                                 PointEval<>       &eval_interface_l,
                                                 PointEval<>       &eval_interface_g,
                                                 DomainEval<>      *T_eval_cell_l,
                                                 DomainEval<>      *T_eval_cell_g,
                                                 PointEval<>       *T_eval_subdomain_l,
                                                 PointEval<>       *T_eval_interface_l,
                                                 PointEval<>       *T_eval_interface_g,
                                                 DomainEval<dim>   *vel_eval,
                                                 PointEval<dim>    *vel_eval_subdomain_l,
                                                 PointEval<dim>    *vel_eval_subdomain_g,
                                                 const bool         do_reinit_cell = true) const;

    void
    tangent_cell_loop(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &dst,
      const VectorType                                                       &src,
      const std::pair<unsigned int, unsigned int>                            &cell_range) const;

    void
    tangent_inner_face_loop(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &dst,
      const VectorType                                                       &src,
      const std::pair<unsigned int, unsigned int>                            &face_range) const;

    void
    tangent_boundary_face_loop(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &dst,
      const VectorType                                                       &src,
      const std::pair<unsigned int, unsigned int>                            &face_range) const;

    /*
     * Local appliers for residual
     */
    void
    residual_cell_loop(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
      VectorType                                                             &residual,
      const VectorType                                                       &temperature_new,
      const std::pair<unsigned int, unsigned int>                            &cell_range) const;

    void
    residual_inner_face_loop(
      const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free_in,
      VectorType                                                             &residual,
      const VectorType                                                       &temperature_new,
      const std::pair<unsigned int, unsigned int>                            &face_range) const;

    void
    residual_boundary_face_loop(
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
