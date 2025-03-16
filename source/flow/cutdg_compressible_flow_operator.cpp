
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_explicit_utils.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/cutdg_compressible_flow_operator.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  template <unsigned int dim, typename number, bool is_viscous>
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::CutDGCompressibleFlowOperator(
    CompressibleFlowScratchData<dim, number> &flow_scratch_data,
    const MappingInfoType                    &mapping_info_surface_in,
    const MappingInfoVectorType              &mapping_info_cells_in,
    const MappingInfoVectorType              &mapping_info_faces_in)
    : flow_scratch_data(flow_scratch_data)
    , convective_terms(flow_scratch_data.flow_data)
    , viscous_terms(flow_scratch_data.flow_data)
    , mapping_info_surface(mapping_info_surface_in)
    , mapping_info_cells(mapping_info_cells_in)
    , mapping_info_faces(mapping_info_faces_in)
    , fe_point_temp(FE_DGQ<dim>(flow_scratch_data.flow_data.fe.degree), dim + 2)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
  {}

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::compute_inverse_time_step(
    const number &time_step)
  {
    Assert(time_step > 0., dealii::ExcMessage("Time step size must be larger than 0!"));
    inv_time_step = 1. / time_step;
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::vmult(VectorType       &dst,
                                                                const VectorType &src) const
  {
    typedef std::function<void(const dealii::MatrixFree<dim, number> &,
                               dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                               const dealii::LinearAlgebra::distributed::Vector<number> &src,
                               const std::pair<unsigned int, unsigned int> &)>
      local_applier_type;

    local_applier_type cell = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_cell_lhs);
    local_applier_type face = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_face_lhs);
    local_applier_type boundary_face =
      MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_boundary_face_lhs);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell,
      face,
      boundary_face,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::create_rhs(const number     &time,
                                                                     const number     &time_step,
                                                                     VectorType       &dst,
                                                                     const VectorType &src) const
  {
    AssertThrow(!dealii::numbers::is_invalid(inv_time_step),
                dealii::ExcMessage(
                  "Inverse time step size must be set to compute the rhs vector."));

    typedef std::function<void(const MatrixFree<dim, number> &,
                               LinearAlgebra::distributed::Vector<number>       &dst,
                               const LinearAlgebra::distributed::Vector<number> &src,
                               const std::pair<unsigned int, unsigned int> &)>
      local_applier_type;

    flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
    local_applier_type cell          = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_cell_rhs);
    local_applier_type face          = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_face_rhs);
    local_applier_type boundary_face = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_boundary_face_rhs);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell,
      face,
      boundary_face,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);

    dst *= time_step;
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::set_inflow_field_unfitted_boundary(
    std::shared_ptr<dealii::Function<dim>> &inflow_function)
  {
    unfitted_inflow = inflow_function;
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::set_unfitted_object_velocity(
    std::shared_ptr<dealii::Function<dim>> &velocity_function)
  {
    unfitted_object_velocity = velocity_function;
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::local_apply_cell_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto active_fe_index =
      flow_scratch_data.scratch_data.get_matrix_free().get_cell_range_category(cell_range);

    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim>                 *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(flow_scratch_data.body_force.get());

    if (constant_function)
      constant_body_force = VectorTools::evaluate_function_at_vectorized_points(
        *constant_function, dealii::Point<dim, dealii::VectorizedArray<number>>());

    if (active_fe_index == CutUtil::CellCategory::liquid)
      {
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src,
                                EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing));

            for (const unsigned int q : phi.quadrature_point_indices())
              {
                auto [flux, grad_flux] =
                  rhs_cell_integral_kernel<dim, number, FECellIntegrator<dim, dim + 2, number>>(
                    phi,
                    q,
                    constant_function ? &constant_body_force : nullptr,
                    convective_terms,
                    viscous_terms,
                    flow_scratch_data);

                // consider mass term
                if (flow_scratch_data.body_force.get() != nullptr)
                  phi.submit_value(flux + phi.get_value(q) * inv_time_step, q);
                else
                  phi.submit_value(phi.get_value(q) * inv_time_step, q);
                phi.submit_gradient(grad_flux, q);
              }

            phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
          }
      }
    else if (active_fe_index == CutUtil::CellCategory::intersected)
      {
        constexpr unsigned int n_lanes = VectorizedArray<number>::size();

        dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>> fe_point_eval(
          *mapping_info_cells[0], fe_point_temp);
        dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
          fe_point_surface(mapping_info_surface, fe_point_temp);

        for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
             ++cell_batch)
          {
            phi.reinit(cell_batch);
            phi.read_dof_values(src);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell_batch);
                 ++lane)
              {
                // evaluate for domain integral
                fe_point_eval.reinit(cell_batch * n_lanes + lane);
                fe_point_eval.evaluate(
                  StridedArrayView<const number, n_lanes>(&phi.begin_dof_values()[0][lane],
                                                          n_dofs_per_cell),
                  EvaluationFlags::values |
                    (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing));

                // evaluate for surface integral
                fe_point_surface.reinit(cell_batch * n_lanes + lane);
                fe_point_surface.evaluate(StridedArrayView<const number, n_lanes>(
                                            &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                          EvaluationFlags::values | EvaluationFlags::gradients);

                // do domain integral
                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  {
                    auto [flux, grad_flux] = rhs_cell_integral_kernel<
                      dim,
                      number,
                      dealii::FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>>(
                      fe_point_eval,
                      q,
                      constant_function ? &constant_body_force : nullptr,
                      convective_terms,
                      viscous_terms,
                      flow_scratch_data);

                    // consider mass term
                    if (flow_scratch_data.body_force.get() != nullptr)
                      fe_point_eval.submit_value(flux + fe_point_eval.get_value(q) * inv_time_step,
                                                 q);
                    else
                      fe_point_eval.submit_value(fe_point_eval.get_value(q) * inv_time_step, q);
                    fe_point_eval.submit_gradient(grad_flux, q);
                  }

                fe_point_eval.integrate(StridedArrayView<number, n_lanes>(
                                          &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                        EvaluationFlags::values | EvaluationFlags::gradients);

                const auto penalty_parameter =
                  is_viscous ? phi.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                               0.;

                // do surface integral
                for (const unsigned int q : fe_point_surface.quadrature_point_indices())
                  {
                    const auto w_m = fe_point_surface.get_value(q);
                    // minus sign is required, as we need a normal which points outside the
                    // active region (the orientation of fe_point_surface.normal_vector(q) depends
                    // on the level set sign of the active region!)
                    const auto normal   = -fe_point_surface.normal_vector(q);
                    const auto grad_w_m = fe_point_surface.get_gradient(q);

                    Tensor<1, dim + 2, VectorizedArray<number>>                 w_p;
                    Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>> grad_w_p;

                    get_adjacent_face_values_at_unfitted_boundary(
                      fe_point_surface.quadrature_point(q), w_m, w_p, grad_w_m, grad_w_p);

                    auto flux =
                      convective_terms.calculate_convective_numerical_flux(w_m, w_p, normal);

                    if (is_viscous)
                      flux -= viscous_terms.calculate_viscous_numerical_flux(
                        w_m, w_p, grad_w_m, grad_w_p, normal, penalty_parameter);

                    fe_point_surface.submit_value(-flux, q);

                    if (is_viscous)
                      {
                        auto numerical_flux_gradient =
                          viscous_terms.calculate_viscous_numerical_flux_gradient(w_m, w_p, normal)
                            .first;

                        fe_point_surface.submit_gradient(numerical_flux_gradient, q);
                      }
                  }

                fe_point_surface.integrate(StridedArrayView<number, n_lanes>(
                                  &phi.begin_dof_values()[0][lane],
                                  n_dofs_per_cell),
                                  EvaluationFlags::values |
                                        (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing), true
                                  /*specify flag 'true' for summing the integrated values into the solution values*/);
              }
            phi.distribute_local_to_global(dst);
          }
      }
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::local_apply_face_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);

    const auto face_category =
      flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);

    if (const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);
        face_type == CutUtil::FaceType::inside_face_liquid or
        face_type == CutUtil::FaceType::mixed_face_liquid)
      {
        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi_p.reinit(face);
            phi_p.gather_evaluate(src,
                                  EvaluationFlags::values |
                                    (is_viscous ? EvaluationFlags::gradients :
                                                  EvaluationFlags::nothing));

            phi_m.reinit(face);
            phi_m.gather_evaluate(src,
                                  EvaluationFlags::values |
                                    (is_viscous ? EvaluationFlags::gradients :
                                                  EvaluationFlags::nothing));

            // factor 0.5 for interior face
            const dealii::VectorizedArray<number> penalty_parameter =
              is_viscous ?
                0.5 * std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                               phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
                0.;

            for (const unsigned int q : phi_m.quadrature_point_indices())
              {
                auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
                  rhs_face_integral_kernel<dim, number, FEFaceIntegrator<dim, dim + 2, number>>(
                    phi_m,
                    phi_p,
                    q,
                    penalty_parameter,
                    convective_terms,
                    viscous_terms);

                phi_m.submit_value(flux_m, q);
                phi_p.submit_value(flux_p, q);
                if (is_viscous)
                  {
                    phi_m.submit_gradient(grad_flux_m, q);
                    phi_p.submit_gradient(grad_flux_p, q);
                  }
              }

            phi_p.integrate_scatter(EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing),
                                    dst);
            phi_m.integrate_scatter(EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing),
                                    dst);
          }
      }
    else if (face_type == CutUtil::FaceType::intersected_face)
      {
        FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval_minus(
          *mapping_info_faces[0], fe_point_temp);
        FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval_plus(
          *mapping_info_faces[0], fe_point_temp);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi_p.reinit(face);
            phi_p.read_dof_values(src);
            phi_m.reinit(face);
            phi_m.read_dof_values(src);

            phi_p.project_to_face(
              EvaluationFlags::values |
              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing));
            phi_m.project_to_face(
              EvaluationFlags::values |
              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing));


            const auto face_info =
              flow_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_face_batch(
                   face);
                 ++lane)
              {
                fe_point_eval_minus.reinit(face_info.cells_interior[lane],
                                           static_cast<int>(face_info.interior_face_no));
                fe_point_eval_plus.reinit(face_info.cells_exterior[lane],
                                          static_cast<int>(face_info.exterior_face_no));

                fe_point_eval_minus.evaluate_in_face(&phi_m.get_scratch_data().begin()[0][lane],
                                                     EvaluationFlags::values |
                                                       (is_viscous ? EvaluationFlags::gradients :
                                                                     EvaluationFlags::nothing));

                fe_point_eval_plus.evaluate_in_face(&phi_p.get_scratch_data().begin()[0][lane],
                                                    EvaluationFlags::values |
                                                      (is_viscous ? EvaluationFlags::gradients :
                                                                    EvaluationFlags::nothing));

                // factor 0.5 for interior face
                const dealii::VectorizedArray<number> penalty_parameter =
                  is_viscous ?
                    0.5 *
                      std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                               phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
                    0.;

                for (const unsigned int q : fe_point_eval_minus.quadrature_point_indices())
                  {
                    auto [flux_m, flux_p, grad_flux_m, grad_flux_p] = rhs_face_integral_kernel<
                      dim,
                      number,
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>>(
                      fe_point_eval_minus,
                      fe_point_eval_minus,
                      q,
                      penalty_parameter,
                      convective_terms,
                      viscous_terms);

                    fe_point_eval_minus.submit_value(flux_m, q);
                    fe_point_eval_plus.submit_value(flux_p, q);
                    if (is_viscous)
                      {
                        fe_point_eval_minus.submit_gradient(grad_flux_m, q);
                        fe_point_eval_plus.submit_gradient(grad_flux_p, q);
                      }
                  }

                fe_point_eval_minus.integrate_in_face(&phi_m.get_scratch_data().begin()[0][lane],
                                                      EvaluationFlags::values |
                                                        (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

                fe_point_eval_plus.integrate_in_face(&phi_p.get_scratch_data().begin()[0][lane],
                                                     EvaluationFlags::values |
                                                       (is_viscous ? EvaluationFlags::gradients :
                                                                     EvaluationFlags::nothing));
              }

            phi_m.collect_from_face(EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing),
                                    phi_m.begin_dof_values());

            phi_p.collect_from_face(EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing),
                                    phi_p.begin_dof_values());

            phi_m.distribute_local_to_global(dst);
            phi_p.distribute_local_to_global(dst);
          }
      }
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::local_apply_boundary_face_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               true,
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    const auto face_category =
      flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);

    if (face_category.first == CutUtil::CellCategory::liquid)
      {
        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi.reinit(face);
            phi.gather_evaluate(src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);

            const dealii::VectorizedArray<number> penalty_parameter =
              is_viscous ? phi.read_cell_data(flow_scratch_data.interior_penalty_parameter) : 0.;

            for (const unsigned int q : phi.quadrature_point_indices())
              {
                auto [flux_m, grad_flux_m] =
                  rhs_boundary_face_integral_kernel<dim,
                                                    number,
                                                    FEFaceIntegrator<dim, dim + 2, number>>(
                    phi,
                    q,
                    flow_scratch_data.scratch_data.get_matrix_free().get_boundary_id(face),
                    penalty_parameter,
                    convective_terms,
                    viscous_terms,
                    flow_scratch_data);
                phi.submit_value(flux_m, q);
                if (is_viscous)
                  phi.submit_gradient(grad_flux_m, q);
              }

            phi.integrate_scatter(EvaluationFlags::values |
                                    (is_viscous ? EvaluationFlags::gradients :
                                                  EvaluationFlags::nothing),
                                  dst);
          }
      }
    else if (face_category.first == CutUtil::CellCategory::intersected)
      {
        FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval_minus(
          *mapping_info_faces[0], fe_point_temp);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi.reinit(face);
            phi.read_dof_values(src);

            phi.project_to_face(EvaluationFlags::values | EvaluationFlags::gradients);

            const auto face_info =
              flow_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_face_batch(
                   face);
                 ++lane)
              {
                fe_point_eval_minus.reinit(face_info.cells_interior[lane],
                                           static_cast<int>(face_info.interior_face_no));

                fe_point_eval_minus.evaluate_in_face(&phi.get_scratch_data().begin()[0][lane],
                                                     EvaluationFlags::values |
                                                       EvaluationFlags::gradients);

                const dealii::VectorizedArray<number> penalty_parameter =
                  is_viscous ? phi.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                               0.;

                for (const unsigned int q : fe_point_eval_minus.quadrature_point_indices())
                  {
                    auto [flux_m, grad_flux_m] = rhs_boundary_face_integral_kernel<
                      dim,
                      number,
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>>(
                      fe_point_eval_minus,
                      q,
                      flow_scratch_data.scratch_data.get_matrix_free().get_boundary_id(face),
                      penalty_parameter,
                      convective_terms,
                      viscous_terms,
                      flow_scratch_data);

                    fe_point_eval_minus.submit_value(flux_m, q);
                    if (is_viscous)
                      fe_point_eval_minus.submit_gradient(grad_flux_m, q);
                  }

                fe_point_eval_minus.integrate_in_face(&phi.get_scratch_data().begin()[0][lane],
                                                      dealii::EvaluationFlags::values |
                                                        dealii::EvaluationFlags::gradients);
              }

            phi.collect_from_face(dealii::EvaluationFlags::values |
                                    dealii::EvaluationFlags::gradients,
                                  phi.begin_dof_values());

            phi.distribute_local_to_global(dst);
          }
      }
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::local_apply_cell_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto active_fe_index =
      flow_scratch_data.scratch_data.get_matrix_free().get_cell_range_category(cell_range);

    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    if (active_fe_index == CutUtil::CellCategory::liquid)
      {
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, dealii::EvaluationFlags::values);

            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_value(phi.get_value(q), q);

            phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
          }
      }
    else if (active_fe_index == CutUtil::CellCategory::intersected)
      {
        constexpr unsigned int n_lanes = VectorizedArray<number>::size();

        FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval(
          *mapping_info_cells[0], fe_point_temp);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values(src);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell);
                 ++lane)
              {
                // evaluate for domain integral
                fe_point_eval.reinit(cell * n_lanes + lane);
                fe_point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                         &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                       EvaluationFlags::values);

                // do domain integral
                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  fe_point_eval.submit_value(fe_point_eval.get_value(q), q);

                fe_point_eval.integrate(StridedArrayView<number, n_lanes>(
                                          &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                        dealii::EvaluationFlags::values);
              }
            phi.distribute_local_to_global(dst);
          }
      }
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::local_apply_face_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    // classify face type of the current range
    const auto face_category =
      flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);
    const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);

    FEFaceIntegrator<dim, dim + 2, number> phi_m(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);

    if (face_type == CutUtil::FaceType::intersected_face or
        face_type == CutUtil::FaceType::mixed_face_liquid)
      {
        EvaluationFlags::EvaluationFlags evaluation_flags =
          dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients |
          ((flow_scratch_data.flow_data.fe.degree == 2) ? dealii::EvaluationFlags::hessians :
                                                          dealii::EvaluationFlags::nothing);

        // Currently, we use homogeneous Cartesian grids
        const number cell_side_length = flow_scratch_data.scratch_data.get_min_cell_size();

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi_p.reinit(face);
            phi_p.gather_evaluate(src, evaluation_flags);

            phi_m.reinit(face);
            phi_m.gather_evaluate(src, evaluation_flags);

            for (const unsigned int q : phi_m.quadrature_point_indices())
              {
                const auto u_minus = phi_m.get_value(q);
                const auto u_plus  = phi_p.get_value(q);

                const auto u_normal_grad_minus = phi_m.get_normal_derivative(q);
                const auto u_normal_grad_plus  = phi_p.get_normal_derivative(q);

                const auto ghost_penalty_term_0 =
                  (u_minus - u_plus) *
                  flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_0 *
                  cell_side_length;

                const auto ghost_penalty_term_1 =
                  (u_normal_grad_minus - u_normal_grad_plus) *
                  flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_1 *
                  Utilities::fixed_power<3>(cell_side_length);

                if (flow_scratch_data.flow_data.fe.degree == 2)
                  {
                    const auto u_normal_hessian_minus = phi_m.get_normal_hessian(q);
                    const auto u_normal_hessian_plus  = phi_p.get_normal_hessian(q);

                    const auto ghost_penalty_term_2 =
                      (u_normal_hessian_minus - u_normal_hessian_plus) *
                      flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_2 *
                      Utilities::fixed_power<5>(cell_side_length);

                    phi_m.submit_normal_hessian(ghost_penalty_term_2, q);
                    phi_p.submit_normal_hessian(-ghost_penalty_term_2, q);
                  }

                phi_m.submit_normal_derivative(ghost_penalty_term_1, q);
                phi_p.submit_normal_derivative(-ghost_penalty_term_1, q);

                phi_m.submit_value(ghost_penalty_term_0, q);
                phi_p.submit_value(-ghost_penalty_term_0, q);
              }

            phi_p.integrate_scatter(evaluation_flags, dst);
            phi_m.integrate_scatter(evaluation_flags, dst);
          }
      }
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::local_apply_boundary_face_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType &,
    const VectorType &,
    const std::pair<unsigned, unsigned> &) const
  {
    // nothing to do here
  }

  template <unsigned int dim, typename number, bool is_viscous>
  void
  CutDGCompressibleFlowOperator<dim, number, is_viscous>::
    get_adjacent_face_values_at_unfitted_boundary(
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      const ConservedVariablesType                              &w_m,
      ConservedVariablesType                                    &w_p,
      const ConservedVariablesGradType                          &grad_w_m,
      ConservedVariablesGradType                                &grad_w_p) const
  {
    if (flow_scratch_data.flow_data.cut.unfitted_flow_boundary_condition == "no_slip_wall")
      {
        // homogeneous Neumann
        w_p[0]      = w_m[0];
        grad_w_p[0] = -grad_w_m[0];
        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            if (unfitted_object_velocity != nullptr)
              w_p[d + 1] = VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                             *unfitted_object_velocity, q_point, d) *
                           w_m[0];
            else // assume non-moving unfitted boundary
              w_p[d + 1] = 0.;

            grad_w_p[d + 1] = grad_w_m[d + 1];
          }
        // homogeneous Neumann
        grad_w_p[dim + 1] = -grad_w_m[dim + 1];
        w_p[dim + 1]      = w_m[dim + 1];
      }
    else if (flow_scratch_data.flow_data.cut.unfitted_flow_boundary_condition == "inflow")
      {
        // Dirichlet
        w_p = VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
          *unfitted_inflow, q_point);
        grad_w_p = grad_w_m;
      }
    else
      AssertThrow(false, dealii::ExcMessage("Unknown boundary type for unfitted boundary."));
  }

  template class CutDGCompressibleFlowOperator<1, double, true>;
  template class CutDGCompressibleFlowOperator<2, double, true>;
  template class CutDGCompressibleFlowOperator<3, double, true>;
  template class CutDGCompressibleFlowOperator<1, double, false>;
  template class CutDGCompressibleFlowOperator<2, double, false>;
  template class CutDGCompressibleFlowOperator<3, double, false>;
} // namespace MeltPoolDG::Flow
