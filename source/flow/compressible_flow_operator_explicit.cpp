
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/flow/compressible_flow_operator_explicit.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>

namespace MeltPoolDG::Flow
{
  template <unsigned int dim, typename number>
  CompressibleFlowOperatorExplicit<dim, number>::CompressibleFlowOperatorExplicit(
    const CompressibleFlowData                     &comp_flow_data_in,
    const ScratchData<dim>                         &scratch_data_in,
    ::TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
    unsigned int                                    comp_flow_dof_idx_in,
    unsigned int                                    comp_flow_quad_idx_in)
    : CompressibleFlowOperatorBase<dim, number>(comp_flow_data_in,
                                                scratch_data_in,
                                                solution_history_in,
                                                comp_flow_dof_idx_in,
                                                comp_flow_quad_idx_in)
  {
    time_integrator =
      std::unique_ptr<TimeIntegratorBase<number, CompressibleFlowOperatorExplicit<dim, number>>>(
        explicit_time_integrator_factory<number, CompressibleFlowOperatorExplicit<dim, number>>(
          this->comp_flow_data.time_integrator, this->scratch_data.get_timer()));
    this->solution_history.resize(time_integrator->required_solution_history_size());
  }

  template <unsigned int dim, typename number>
  void
  CompressibleFlowOperatorExplicit<dim, number>::advance_time_step(
    number                                                        current_time,
    number                                                        time_step,
    std::function<void(number, VectorType &, const VectorType &)> pre_processing,
    std::function<void(number, VectorType &, const VectorType &)> post_processing)
  {
    time_integrator->perform_time_step(
      *this, current_time, time_step, this->solution_history, pre_processing, post_processing);
  }


  template <unsigned int dim, typename number>
  void
  CompressibleFlowOperatorExplicit<dim, number>::reinit()
  {
    time_integrator->reinit(this->solution_history);
    this->calculate_interior_penalty_parameter();
  }


  template <unsigned int dim, typename number>
  void
  CompressibleFlowOperatorExplicit<dim, number>::apply_operator(
    number,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    typedef std::function<void(const MatrixFree<dim, number> &,
                               LinearAlgebra::distributed::Vector<number>       &dst,
                               const LinearAlgebra::distributed::Vector<number> &src,
                               const std::pair<unsigned int, unsigned int> &)>
      local_applier_type;

    local_applier_type cell          = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_cell);
    local_applier_type face          = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_face);
    local_applier_type boundary_face = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_boundary_face);
    this->scratch_data.get_matrix_free().loop(cell, face, boundary_face, dst, src, false);

    local_applier_type inverse = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_inverse_mass_matrix);
    this->scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }


  template <unsigned int dim, typename number>
  void
  CompressibleFlowOperatorExplicit<dim, number>::local_apply_cell(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(this->scratch_data.get_matrix_free(),
                                               this->comp_flow_dof_idx,
                                               this->comp_flow_quad_idx);
    FECellIntegrator<dim, dim + 2, number> fe_user_rhs(this->scratch_data.get_matrix_free(),
                                                       this->comp_flow_dof_idx,
                                                       this->comp_flow_quad_idx);

    Tensor<1, dim, VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim> *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(this->body_force.get());

    if (constant_function)
      constant_body_force =
        VectorTools::evaluate_function_at_vectorized_points(*constant_function,
                                                            Point<dim, VectorizedArray<number>>());

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src,
                            EvaluationFlags::values |
                              ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                 EvaluationFlags::gradients :
                                 EvaluationFlags::nothing));
        fe_user_rhs.reinit(cell);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            auto [flux, grad_flux] =
              this->template rhs_cell_intergal_kernel<FECellIntegrator<dim, dim + 2, number>>(
                phi, q, constant_function ? &constant_body_force : nullptr);

            if (this->body_force.get() != nullptr)
              phi.submit_value(flux, q);
            phi.submit_gradient(grad_flux, q);
          }

        phi.integrate_scatter(((this->body_force.get() != nullptr) ? EvaluationFlags::values :
                                                                     EvaluationFlags::nothing) |
                                EvaluationFlags::gradients,
                              dst);
      }
  }

  template <unsigned int dim, typename number>
  void
  CompressibleFlowOperatorExplicit<dim, number>::local_apply_face(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(this->scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 this->comp_flow_dof_idx,
                                                 this->comp_flow_quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(this->scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 this->comp_flow_dof_idx,
                                                 this->comp_flow_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src,
                              EvaluationFlags::values |
                                ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(src,
                              EvaluationFlags::values |
                                ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing));

        const VectorizedArray<number> penalty_parameter =
          (this->comp_flow_data.dynamic_viscosity > 0) ?
            std::max(phi_m.read_cell_data(this->interior_penalty_parameter),
                     phi_p.read_cell_data(this->interior_penalty_parameter)) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
              this->template rhs_face_integral_kernel<FEFaceIntegrator<dim, dim + 2, number>>(
                phi_m, phi_p, q, penalty_parameter);

            phi_m.submit_value(flux_m, q);
            phi_p.submit_value(flux_p, q);
            if (this->comp_flow_data.dynamic_viscosity > 0)
              {
                phi_m.submit_gradient(grad_flux_m, q);
                phi_p.submit_gradient(grad_flux_p, q);
              }
          }

        phi_p.integrate_scatter(EvaluationFlags::values |
                                  ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
        phi_m.integrate_scatter(EvaluationFlags::values |
                                  ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
      }
  }

  template <unsigned int dim, typename number>
  void
  CompressibleFlowOperatorExplicit<dim, number>::local_apply_boundary_face(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi(this->scratch_data.get_matrix_free(),
                                               true,
                                               this->comp_flow_dof_idx,
                                               this->comp_flow_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> penalty_parameter =
          (this->comp_flow_data.dynamic_viscosity > 0) ?
            phi.read_cell_data(this->interior_penalty_parameter) :
            0.;

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            auto [flux_m, grad_flux_m] = this->template rhs_boundary_face_integral_kernel<
              FEFaceIntegrator<dim, dim + 2, number>>(phi, q, penalty_parameter);

            phi.submit_value(flux_m, q);
            if (this->comp_flow_data.dynamic_viscosity > 0)
              phi.submit_gradient(grad_flux_m, q);
          }

        phi.integrate_scatter(EvaluationFlags::values |
                                ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing),
                              dst);
      }
  }


  template class CompressibleFlowOperatorExplicit<1, double>;
  template class CompressibleFlowOperatorExplicit<2, double>;
  template class CompressibleFlowOperatorExplicit<3, double>;
} // namespace MeltPoolDG::Flow