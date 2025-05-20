#include <meltpooldg/flow/compressible_flow_explicit_utils.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_explicit.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>


namespace MeltPoolDG::Flow
{
  using namespace dealii;
  template <unsigned int dim, typename number, bool is_viscous>
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::DGCompressibleFlowOperatorExplicit(
    CompressibleFlowScratchData<dim, number>                   &flow_scratch_data,
    std::unique_ptr<ExplicitExternalFluidForces<dim, number>> &&external_forces)
    : flow_scratch_data(flow_scratch_data)
    , convective_terms(flow_scratch_data.flow_data)
    , viscous_terms(flow_scratch_data.flow_data)
    , external_forces(std::move(external_forces))
  {}

  template <unsigned int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::reinit()
  {
    // nothing to do here
  }

  template <unsigned int dim, typename number, bool is_viscous>
  std::unique_ptr<TimeIntegration::TimeIntegratorBase<number>>
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::
    make_application_specific_time_integrator(
      const TimeIntegration::TimeIntegratorData<number> &time_integrator_data)
  {
    return std::unique_ptr<TimeIntegration::TimeIntegratorBase<number>>(
      explicit_time_integrator_factory<number,
                                       DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>>(
        *this, time_integrator_data, flow_scratch_data.scratch_data.get_timer()));
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::apply_operator(
    const number                                           time,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(this->local_apply_cell);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell, face, boundary_face, dst, src, false);

    local_applier_type inverse =
      [dof_idx = flow_scratch_data.dof_idx,
       quad_idx =
         flow_scratch_data.quad_idx](const MatrixFree<dim, number>                    &matrix_free,
                                     LinearAlgebra::distributed::Vector<number>       &dst,
                                     const LinearAlgebra::distributed::Vector<number> &src,
                                     const std::pair<unsigned int, unsigned int>       cell_range) {
        Utilities::MatrixFree::local_apply_inverse_mass_matrix<dim, dim + 2, number>(
          matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };
    flow_scratch_data.scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::local_apply_cell(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim>                 *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(flow_scratch_data.body_force.get());

    if (constant_function)
      constant_body_force = VectorTools::evaluate_function_at_vectorized_points(
        *constant_function, dealii::Point<dim, dealii::VectorizedArray<number>>());

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src,
                            EvaluationFlags::values |
                              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing));

        if (external_forces != nullptr)
          {
            external_forces->cell_operation(flow_scratch_data.scratch_data.get_matrix_free(),
                                            cell,
                                            phi.n_lanes);
          }

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            auto [flux, grad_flux] =
              rhs_cell_integral_kernel<dim,
                                       number,
                                       FECellIntegrator<dim, dim + 2, number>,
                                       is_viscous>(phi,
                                                   q,
                                                   constant_function ? &constant_body_force :
                                                                       nullptr,
                                                   convective_terms,
                                                   viscous_terms,
                                                   flow_scratch_data);

            if (external_forces != nullptr)
              {
                const ConservedVariablesType w_q = phi.get_value(q);
                flux += external_forces->quad_operation(phi.quadrature_point(q), w_q);
              }

            if (flow_scratch_data.body_force.get() != nullptr || external_forces != nullptr)
              phi.submit_value(flux, q);
            phi.submit_gradient(grad_flux, q);
          }

        phi.integrate_scatter(((flow_scratch_data.body_force.get() != nullptr ||
                                external_forces != nullptr) ?
                                 EvaluationFlags::values :
                                 EvaluationFlags::nothing) |
                                EvaluationFlags::gradients,
                              dst);
      }
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::local_apply_face(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 true,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 false,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src,
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(src,
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous ?
            std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                     phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
              rhs_face_integral_kernel<dim,
                                       number,
                                       FEFaceIntegrator<dim, dim + 2, number>,
                                       is_viscous>(
                phi_m, phi_p, q, interior_penalty_parameter, convective_terms, viscous_terms);

            phi_m.submit_value(flux_m, q);
            phi_p.submit_value(flux_p, q);
            if (is_viscous)
              {
                phi_m.submit_gradient(grad_flux_m, q);
                phi_p.submit_gradient(grad_flux_p, q);
              }
          }

        phi_p.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing),
                                dst);
        phi_m.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing),
                                dst);
      }
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::local_apply_boundary_face(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               true,
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous ? phi.read_cell_data(flow_scratch_data.interior_penalty_parameter) : 0.;

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            auto [flux_m, grad_flux_m] =
              rhs_boundary_face_integral_kernel<dim,
                                                number,
                                                FEFaceIntegrator<dim, dim + 2, number>,
                                                is_viscous>(phi,
                                                            q,
                                                            phi.boundary_id(),
                                                            interior_penalty_parameter,
                                                            convective_terms,
                                                            viscous_terms,
                                                            flow_scratch_data);

            phi.submit_value(flux_m, q);
            if (is_viscous)
              phi.submit_gradient(grad_flux_m, q);
          }

        phi.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing),
                              dst);
      }
  }

  template class DGCompressibleFlowOperatorExplicit<1, double, true>;
  template class DGCompressibleFlowOperatorExplicit<2, double, true>;
  template class DGCompressibleFlowOperatorExplicit<3, double, true>;
  template class DGCompressibleFlowOperatorExplicit<1, double, false>;
  template class DGCompressibleFlowOperatorExplicit<2, double, false>;
  template class DGCompressibleFlowOperatorExplicit<3, double, false>;
} // namespace MeltPoolDG::Flow
