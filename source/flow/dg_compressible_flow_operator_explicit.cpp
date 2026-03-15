#include <meltpooldg/flow/compressible_flow_explicit_utils.hpp>
#include <meltpooldg/flow/compressible_flow_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/flow/compressible_flow_views.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_explicit.hpp>
#include <meltpooldg/flow/dg_generic_convection_diffusion_worker.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>


namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim, typename number, bool is_viscous>
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::DGCompressibleFlowOperatorExplicit(
    CompressibleFlowScratchData<dim, number> &flow_scratch_data)
    : flow_scratch_data(flow_scratch_data)
    , time_integrator(flow_scratch_data.flow_data.time_integrator)
    , generic_operator({flow_scratch_data.scratch_data.get_matrix_free(),
                        flow_scratch_data.dof_idx,
                        flow_scratch_data.quad_idx})
  {
    time_integrator.configure_rhs(
      std::bind_front(&DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::apply_operator,
                      this));

    dealii::EvaluationFlags::EvaluationFlags face_and_cell_eval_flags =
      EvaluationFlags::values |
      (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing);

    generic_operator.add_cell_eval_flags(face_and_cell_eval_flags, EvaluationFlags::gradients);

    generic_operator.add_face_eval_flags(face_and_cell_eval_flags,
                                         EvaluationFlags::values |
                                           (is_viscous ? EvaluationFlags::gradients :
                                                         EvaluationFlags::nothing));

    generic_operator.add_cell_quadrature_operation(std::bind_front(
      &DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::cell_quad_operation, this));
    generic_operator.add_inner_face_quadrature_operation(std::bind_front(
      &DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::face_quad_operation, this));
    generic_operator.add_boundary_face_quadrature_operation(std::bind_front(
      &DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::boundary_face_quad_operation,
      this));
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::reinit()
  {
    flow_scratch_data.reinit(time_integrator.required_solution_history_size());
    time_integrator.reinit(flow_scratch_data.solution_history.get_current_solution());
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::advance_time_step(number time,
                                                                                 number time_step)
  {
    std::function<void(number, number, VectorType &, const VectorType &)> pre_processing =
      [&](number time, number, VectorType &, const VectorType &) -> void {
      flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
    };

    time_integrator.perform_time_step(
      time,
      time_step,
      flow_scratch_data.solution_history,
      pre_processing,
      std::function<void(number, number, VectorType &, const VectorType &)>());
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::apply_operator(
    const number                                           time,
    const number                                           time_step,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    current_time_step = time_step;
    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    flow_scratch_data.boundary_conditions.update_boundary_conditions(time);

    generic_operator.matrix_free_loop(dst, src, false);

    local_applier_type inverse =
      [dof_idx = flow_scratch_data.dof_idx,
       quad_idx =
         flow_scratch_data.quad_idx](const MatrixFree<dim, number>                    &matrix_free,
                                     LinearAlgebra::distributed::Vector<number>       &dst,
                                     const LinearAlgebra::distributed::Vector<number> &src,
                                     const std::pair<unsigned int, unsigned int>      &cell_range) {
        Utilities::MatrixFree::local_apply_inverse_mass_matrix<dim, dim + 2, number>(
          matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };
    flow_scratch_data.scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::add_external_force(
    std::shared_ptr<ExternalFlowForce<dim, number>> external_force,
    std::shared_ptr<ExternalFlowForceJacobian<dim, number>>)
  {
    Assert(external_force != nullptr, dealii::ExcInternalError());
    external_forces.push_back(external_force);
    generic_operator.add_cell_eval_flags(EvaluationFlags::nothing, EvaluationFlags::values);
  }

  template <int dim, typename number, bool is_viscous>
  std::tuple<CompressibleFlow::SourceType<dim, number>, CompressibleFlow::FluxType<dim, number>>
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::cell_quad_operation(
    const unsigned int                                                   cell_batch_index,
    const CompressibleFlow::ConservedVariablesType<dim, number>         &w_q,
    const CompressibleFlow::ConservedVariablesGradientType<dim, number> &grad_w_q,
    const dealii::Point<dim, dealii::VectorizedArray<number>>           &quadrature_point) const
  {
    CompressibleFlow::SourceType<dim, number> source;
    CompressibleFlow::FluxType<dim, number>   flux;

    std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cells =
      cells_in_cell_batch(flow_scratch_data.scratch_data.get_matrix_free(), cell_batch_index);

    if (is_viscous)
      flux = ConvectionDiffusionOperator::cell(
        w_q,
        grad_w_q,
        CompressibleConvectiveFlux<dim, number>(flow_scratch_data.material.data),
        CompressibleDiffusiveFlux<dim, number>(flow_scratch_data.material.data));
    else
      flux = ConvectionOperator::cell(
        w_q, CompressibleConvectiveFlux<dim, number>(flow_scratch_data.material.data));

    for (auto &external_force : external_forces)
      source += external_force->value(current_time_step, cells, quadrature_point, w_q);

    return {source, flux};
  }

  template <int dim, typename number, bool is_viscous>
  std::tuple<CompressibleFlow::FaceFluxType<dim, number>,
             CompressibleFlow::FaceGradientFluxType<dim, number>,
             CompressibleFlow::FaceFluxType<dim, number>,
             CompressibleFlow::FaceGradientFluxType<dim, number>>
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::face_quad_operation(
    const std::array<unsigned int, VectorizedArrayType::size()>         &cell_ids_m,
    const std::array<unsigned int, VectorizedArrayType::size()>         &cell_ids_p,
    const CompressibleFlow::ConservedVariablesType<dim, number>         &w_m,
    const CompressibleFlow::ConservedVariablesGradientType<dim, number> &grad_w_m,
    const CompressibleFlow::ConservedVariablesType<dim, number>         &w_p,
    const CompressibleFlow::ConservedVariablesGradientType<dim, number> &grad_w_p,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>       &normal,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &) const
  {
    VectorizedArrayType interior_penalty_parameter(0.);
    if (is_viscous)
      {
        VectorizedArrayType penalty_parameter_m(0.);
        VectorizedArrayType penalty_parameter_p(0.);
        matrix_free_read_cell_data(cell_ids_m,
                                   flow_scratch_data.interior_penalty_parameter,
                                   penalty_parameter_m);
        matrix_free_read_cell_data(cell_ids_p,
                                   flow_scratch_data.interior_penalty_parameter,
                                   penalty_parameter_p);
        interior_penalty_parameter = flow_scratch_data.material.data.dynamic_viscosity /
                                     flow_scratch_data.material.data.reference_density *
                                     std::max(penalty_parameter_m, penalty_parameter_p);
      }

    CompressibleFlow::FaceFluxType<dim, number> flux_m;
    CompressibleFlow::FaceFluxType<dim, number> flux_p;

    CompressibleFlow::FaceGradientFluxType<dim, number> gradient_flux_m;
    CompressibleFlow::FaceGradientFluxType<dim, number> gradient_flux_p;

    if (is_viscous)
      {
        const auto flux = ConvectionDiffusionOperator::face(
          w_m,
          w_p,
          grad_w_m,
          grad_w_p,
          normal,
          interior_penalty_parameter,
          CompressibleConvectiveFlux<dim, number>(flow_scratch_data.material.data),
          CompressibleDiffusiveFlux<dim, number>(flow_scratch_data.material.data));

        flux_m = flux.inner_face_value;
        flux_p = flux.outer_face_value;

        gradient_flux_m = flux.inner_face_gradient;
        gradient_flux_p = flux.outer_face_gradient;
      }
    else
      {
        const auto flux = ConvectionOperator::face(w_m,
                                                   w_p,
                                                   normal,
                                                   CompressibleConvectiveFlux<dim, number>(
                                                     flow_scratch_data.material.data));

        flux_m = flux.inner_face_value;
        flux_p = flux.outer_face_value;
      }

    return {flux_m, gradient_flux_m, flux_p, gradient_flux_p};
  }

  template <int dim, typename number, bool is_viscous>
  std::tuple<CompressibleFlow::FaceFluxType<dim, number>,
             CompressibleFlow::FaceGradientFluxType<dim, number>>
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::boundary_face_quad_operation(
    const std::array<unsigned int, dealii::VectorizedArray<number>::size()> &cell_ids_m,
    const CompressibleFlow::ConservedVariablesType<dim, number>             &w_m,
    const CompressibleFlow::ConservedVariablesGradientType<dim, number>     &grad_w_m,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>           &normal,
    const dealii::Point<dim, dealii::VectorizedArray<number>>               &quadrature_point,
    const dealii::types::boundary_id                                         boundary_id) const
  {
    using DofValueAndGradientStateViewType = CompressibleFlow::DofValueAndGradientStateView<
      dim,
      number,
      const CompressibleFlow::ConservedVariablesType<dim, number>,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number>>;

    VectorizedArray<number> interior_penalty_parameter(0.);
    if (is_viscous)
      {
        matrix_free_read_cell_data(cell_ids_m,
                                   flow_scratch_data.interior_penalty_parameter,
                                   interior_penalty_parameter);
        interior_penalty_parameter *= flow_scratch_data.material.data.dynamic_viscosity /
                                      flow_scratch_data.material.data.reference_density;
      }

    const auto [w_p, grad_w_p] =
      flow_scratch_data.boundary_conditions.get_boundary_face_value_and_gradient(
        quadrature_point,
        normal,
        boundary_id,
        DofValueAndGradientStateViewType(w_m, grad_w_m, flow_scratch_data.material.data));

    CompressibleFlow::FaceFluxType<dim, number>         flux_m;
    CompressibleFlow::FaceGradientFluxType<dim, number> gradient_flux_m;

    if (is_viscous)
      {
        const auto flux = ConvectionDiffusionOperator::face(
          w_m,
          w_p,
          grad_w_m,
          grad_w_p,
          normal,
          interior_penalty_parameter,
          CompressibleConvectiveFlux<dim, number>(flow_scratch_data.material.data),
          CompressibleDiffusiveFlux<dim, number>(flow_scratch_data.material.data));

        flux_m          = flux.inner_face_value;
        gradient_flux_m = flux.inner_face_gradient;
      }
    else
      {
        const auto flux = ConvectionOperator::face(w_m,
                                                   w_p,
                                                   normal,
                                                   CompressibleConvectiveFlux<dim, number>(
                                                     flow_scratch_data.material.data));

        flux_m = flux.inner_face_value;
      }

    return {flux_m, gradient_flux_m};
  }

  template class DGCompressibleFlowOperatorExplicit<1, double, true>;
  template class DGCompressibleFlowOperatorExplicit<2, double, true>;
  template class DGCompressibleFlowOperatorExplicit<3, double, true>;
  template class DGCompressibleFlowOperatorExplicit<1, double, false>;
  template class DGCompressibleFlowOperatorExplicit<2, double, false>;
  template class DGCompressibleFlowOperatorExplicit<3, double, false>;
} // namespace MeltPoolDG::Flow
