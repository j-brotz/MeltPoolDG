
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_explicit_utils.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_multiphase/compressible_multiphase_interface_kernels.hpp>
#include <meltpooldg/flow/compressible_multiphase/compressible_multiphase_operator.hpp>

namespace MeltPoolDG::Multiphase
{
  using namespace dealii;

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    CompressibleMultiphaseOperator(
      MeltPoolDG::Flow::CompressibleFlowScratchData<dim, number> &flow_scratch_data,
      const MappingInfoType                                      &mapping_info_interface_in,
      const MappingInfoVectorType                                &mapping_info_cells_in,
      const MappingInfoVectorType                                &mapping_info_faces_in)
    : flow_scratch_data(flow_scratch_data)
    , convective_terms_liquid(flow_scratch_data.flow_data)
    , convective_terms_gas(flow_scratch_data.flow_data)
    , viscous_terms_liquid(flow_scratch_data.flow_data)
    , viscous_terms_gas(flow_scratch_data.flow_data)
    , mapping_info_interface(mapping_info_interface_in)
    , mapping_info_cells(mapping_info_cells_in)
    , mapping_info_faces(mapping_info_faces_in)
    , fe_point_temp(FE_DGQ<dim>(flow_scratch_data.flow_data.fe.degree), dim + 2)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
  {
    const double q_1 = 2. * flow_scratch_data.flow_data.material.liquid.dynamic_viscosity +
                       flow_scratch_data.flow_data.material.liquid.thermal_conductivity /
                         (flow_scratch_data.flow_data.material.liquid.specific_isobaric_heat /
                          flow_scratch_data.flow_data.material.liquid.gamma);

    const double q_2 = 2. * flow_scratch_data.flow_data.material.gas.dynamic_viscosity +
                       flow_scratch_data.flow_data.material.gas.thermal_conductivity /
                         (flow_scratch_data.flow_data.material.gas.specific_isobaric_heat /
                          flow_scratch_data.flow_data.material.gas.gamma);

    visc_ave_weight_phase_1 = q_1 / (q_1 + q_2);
    visc_ave_weight_phase_2 = 1. - visc_ave_weight_phase_1;
  }

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(this->local_apply_cell_lhs);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face_lhs);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face_lhs);
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

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::create_rhs(
    const number     &time,
    const number     &time_step,
    VectorType       &dst,
    const VectorType &src) const
  {
    Assert(time_step > 0., dealii::ExcMessage("Time step size must be larger than 0!"));
    inv_time_step = 1. / time_step;

    typedef std::function<void(const MatrixFree<dim, number> &,
                               LinearAlgebra::distributed::Vector<number>       &dst,
                               const LinearAlgebra::distributed::Vector<number> &src,
                               const std::pair<unsigned int, unsigned int> &)>
      local_applier_type;

    flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(local_apply_cell_rhs);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(local_apply_face_rhs);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(local_apply_boundary_face_rhs);
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

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_cell_rhs(const dealii::MatrixFree<dim, number> &,
                         VectorType                          &dst,
                         const VectorType                    &src,
                         const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto cell_category = flow_scratch_data.scratch_data.get_cell_range_category(cell_range);

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim>                 *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(flow_scratch_data.body_force.get());

    if (constant_function)
      constant_body_force = VectorTools::evaluate_function_at_vectorized_points(
        *constant_function, dealii::Point<dim, dealii::VectorizedArray<number>>());

    // lambda function for cell integral
    auto process_cell =
      [&]<bool is_gas_phase, bool is_viscous, typename IntegratorType>(IntegratorType &eval,
                                                                       const auto &convective_terms,
                                                                       const auto &viscous_terms) {
        for (const unsigned int q : eval.quadrature_point_indices())
          {
            const auto [flux, grad_flux] =
              Flow::rhs_cell_integral_kernel<dim, number, IntegratorType, is_viscous>(
                eval,
                q,
                constant_function ? &constant_body_force : nullptr,
                convective_terms,
                viscous_terms,
                flow_scratch_data);

            // consider mass term
            if (flow_scratch_data.body_force.get() != nullptr)
              eval.submit_value(flux + eval.get_value(q) * inv_time_step, q);
            else
              eval.submit_value(eval.get_value(q) * inv_time_step, q);
            eval.submit_gradient(grad_flux, q);
          }
      };

    switch (cell_category)
      {
          case CutUtil::CellCategory::liquid: {
            auto eval_liquid = create_cell_integrator(CutUtil::CellCategory::liquid, 0);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                eval_liquid.reinit(cell);
                eval_liquid.gather_evaluate(src,
                                            EvaluationFlags::values |
                                              (is_viscous_liquid ? EvaluationFlags::gradients :
                                                                   EvaluationFlags::nothing));
                process_cell.template operator()<false, is_viscous_liquid>(eval_liquid,
                                                                           convective_terms_liquid,
                                                                           viscous_terms_liquid);
                eval_liquid.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                              dst);
              }
            break;
          }
          case CutUtil::CellCategory::gas: {
            auto eval_gas = create_cell_integrator(CutUtil::CellCategory::gas, dim + 2);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                eval_gas.reinit(cell);
                eval_gas.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           (is_viscous_gas ? EvaluationFlags::gradients :
                                                             EvaluationFlags::nothing));
                process_cell.template operator()<true, is_viscous_gas>(eval_gas,
                                                                       convective_terms_gas,
                                                                       viscous_terms_gas);
                eval_gas.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                           dst);
              }
            break;
          }
          case CutUtil::CellCategory::intersected: {
            constexpr unsigned int n_lanes = VectorizedArray<number>::size();

            auto eval_liquid_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, 0);
            auto eval_gas_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, dim + 2);

            dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
              eval_point_liquid(*mapping_info_cells[0], fe_point_temp);
            dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
              eval_point_interface_liquid(mapping_info_interface, fe_point_temp);
            dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
              eval_point_gas(*mapping_info_cells[1], fe_point_temp);
            dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
              eval_point_interface_gas(mapping_info_interface, fe_point_temp);

            // reset values for interface velocity
            level_set_advection_operator.clear_interface_velocity();

            for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
                 ++cell_batch)
              {
                eval_liquid_intersected.reinit(cell_batch);
                eval_liquid_intersected.read_dof_values(src);

                eval_gas_intersected.reinit(cell_batch);
                eval_gas_intersected.read_dof_values(src);

                for (unsigned int lane = 0; lane < flow_scratch_data.scratch_data.get_matrix_free()
                                                     .n_active_entries_per_cell_batch(cell_batch);
                     ++lane)
                  {
                    // evaluate for domain integral in liquid phase
                    eval_point_liquid.reinit(cell_batch * n_lanes + lane);
                    eval_point_liquid.evaluate(
                      StridedArrayView<const number, n_lanes>(
                        &eval_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values | (is_viscous_liquid ? EvaluationFlags::gradients :
                                                                     EvaluationFlags::nothing));

                    // evaluate for interface integral in liquid phase
                    eval_point_interface_liquid.reinit(cell_batch * n_lanes + lane);
                    eval_point_interface_liquid.evaluate(
                      StridedArrayView<const number, n_lanes>(
                        &eval_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values | EvaluationFlags::gradients);

                    // evaluate for domain integral in gas phase
                    eval_point_gas.reinit(cell_batch * n_lanes + lane);
                    eval_point_gas.evaluate(
                      StridedArrayView<const number, n_lanes>(
                        &eval_gas_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values |
                        (is_viscous_gas ? EvaluationFlags::gradients : EvaluationFlags::nothing));

                    // evaluate for interface integral in gas phase
                    eval_point_interface_gas.reinit(cell_batch * n_lanes + lane);
                    eval_point_interface_gas.evaluate(
                      StridedArrayView<const number, n_lanes>(
                        &eval_gas_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values | EvaluationFlags::gradients);

                    // do domain integral in liquid phase
                    process_cell.template operator()<false, is_viscous_liquid>(
                      eval_point_liquid, convective_terms_liquid, viscous_terms_liquid);

                    eval_point_liquid.integrate(
                      StridedArrayView<number, n_lanes>(
                        &eval_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values | EvaluationFlags::gradients);

                    // do domain integral in gas phase
                    process_cell.template operator()<true, is_viscous_gas>(eval_point_gas,
                                                                           convective_terms_gas,
                                                                           viscous_terms_gas);

                    eval_point_gas.integrate(StridedArrayView<number, n_lanes>(
                                               &eval_gas_intersected.begin_dof_values()[0][lane],
                                               n_dofs_per_cell),
                                             EvaluationFlags::values | EvaluationFlags::gradients);

                    // do interface integral
                    if (flow_scratch_data.flow_data.interface_jump_conditions.type ==
                        InterfaceNumericalMethod::penalty)
                      {
                        // enumeration for conserved variables component indices
                        using Idx = std::conditional_t<
                          dim == 1,
                          Flow::Idx1D,
                          std::conditional_t<dim == 2,
                                             Flow::Idx2D,
                                             std::conditional_t<dim == 3, Flow::Idx3D, void>>>;

                        for (const unsigned int q :
                             eval_point_interface_liquid.quadrature_point_indices())
                          {
                            auto w_liquid      = eval_point_interface_liquid.get_value(q);
                            auto w_gas         = eval_point_interface_gas.get_value(q);
                            auto grad_w_liquid = eval_point_interface_liquid.get_gradient(q);
                            auto grad_w_gas    = eval_point_interface_gas.get_gradient(q);

                            const auto [flux_liquid, flux_gas] =
                              calculate_convective_and_viscous_interface_flux_penalty<
                                dim,
                                number,
                                ConservedVariablesType,
                                ConservedVariablesGradType>(w_liquid,
                                                            w_gas,
                                                            grad_w_liquid,
                                                            grad_w_gas,
                                                            flow_scratch_data.flow_data,
                                                            viscous_terms_liquid,
                                                            viscous_terms_gas);

                            // Compute the velocity at the interface using the difference in
                            // momentum and density between the liquid and gas phases
                            // (mass conservation across the interface).
                            // indices: [conserved variables component][vectorization index]

                            // Outwards pointing normal vector with respect to the liquid domain.
                            const auto normal = -eval_point_interface_liquid.normal_vector(q);

                            const number interface_velocity =
                              ((w_liquid[Idx::density][0] - w_gas[Idx::density][0]) > 1.e-10 ?
                                 (w_liquid[Idx::momentum_x][0] - w_gas[Idx::momentum_x][0]) /
                                   (w_liquid[Idx::density][0] - w_gas[Idx::density][0]) :
                                 w_liquid[Idx::momentum_x][0] / w_liquid[Idx::density][0]) *
                              normal[0][0];

                            level_set_advection_operator.set_interface_velocity(interface_velocity);

                            eval_point_interface_liquid.submit_value(-flux_liquid, q);
                            eval_point_interface_gas.submit_value(-flux_gas, q);
                          }
                      }
                    else if (flow_scratch_data.flow_data.interface_jump_conditions.type ==
                               InterfaceNumericalMethod::HLLP0_and_SIPG or
                             flow_scratch_data.flow_data.interface_jump_conditions.type ==
                               InterfaceNumericalMethod::HLLP0_and_penalty)
                      {
                        for (const unsigned int q :
                             eval_point_interface_gas.quadrature_point_indices())
                          {
                            const auto w_liquid      = eval_point_interface_liquid.get_value(q);
                            const auto w_gas         = eval_point_interface_gas.get_value(q);
                            const auto grad_w_liquid = eval_point_interface_liquid.get_gradient(q);
                            const auto grad_w_gas    = eval_point_interface_gas.get_gradient(q);
                            // Outwards pointing normal vector with respect to the liquid domain.
                            // (The sign depends on the level-set orientation. Currently, we use
                            // a positive level-set for the liquid phase.)
                            const auto normal = -eval_point_interface_liquid.normal_vector(q);

                            const auto [riemann_flux_liquid,
                                        riemann_flux_gas,
                                        velocity_interface_vec] =
                              calculate_convective_interface_flux_HLLP0<dim,
                                                                        number,
                                                                        ConservedVariablesType,
                                                                        ConservedVariablesGradType>(
                                w_liquid,
                                w_gas,
                                normal,
                                convective_terms_liquid,
                                convective_terms_gas,
                                flow_scratch_data.flow_data);

                            ConservedVariablesType flux_liquid =
                              contract_tensor_with_vector<dim + 2, dim, number>(riemann_flux_liquid,
                                                                                normal);
                            ConservedVariablesType flux_gas =
                              contract_tensor_with_vector<dim + 2, dim, number>(riemann_flux_gas,
                                                                                -normal);
                            // TODO: consider more complex data structure for velocity for dim>1
                            // TODO: project interface velocity in normal direction for dim>1
                            // returns velocity with respect to the outward liquid phase pointing
                            // normal!
                            level_set_advection_operator.set_interface_velocity(
                              velocity_interface_vec[0] * normal[0][0]);

                            if (is_viscous_liquid or is_viscous_gas)
                              {
                                if (flow_scratch_data.flow_data.interface_jump_conditions.type ==
                                    InterfaceNumericalMethod::HLLP0_and_SIPG)
                                  {
                                    const auto [viscous_interface_flux_liquid,
                                                viscous_interface_flux_gas] =
                                      calculate_viscous_interface_flux<dim,
                                                                       number,
                                                                       ConservedVariablesType,
                                                                       ConservedVariablesGradType>(
                                        w_liquid,
                                        w_gas,
                                        grad_w_liquid,
                                        grad_w_gas,
                                        normal,
                                        visc_ave_weight_phase_1,
                                        visc_ave_weight_phase_2,
                                        flow_scratch_data.flow_data.interface_jump_conditions
                                          .hllp0_and_sipg.interior_penalty_parameter_interface,
                                        viscous_terms_liquid,
                                        viscous_terms_gas,
                                        flow_scratch_data.flow_data,
                                        flow_scratch_data.scratch_data.get_min_cell_size());

                                    flux_liquid -= viscous_interface_flux_liquid;
                                    // opposite normal direction for phase 2
                                    flux_gas += viscous_interface_flux_gas;
                                  }
                                else if (flow_scratch_data.flow_data.interface_jump_conditions
                                           .type == InterfaceNumericalMethod::HLLP0_and_penalty)
                                  {
                                    const auto [viscous_interface_flux_liquid,
                                                viscous_interface_flux_gas] =
                                      calculate_viscous_interface_flux_method_3<
                                        dim,
                                        number,
                                        ConservedVariablesType,
                                        ConservedVariablesGradType>(
                                        w_liquid,
                                        w_gas,
                                        grad_w_liquid,
                                        grad_w_gas,
                                        normal,
                                        visc_ave_weight_phase_1,
                                        visc_ave_weight_phase_2,
                                        viscous_terms_liquid,
                                        viscous_terms_gas,
                                        flow_scratch_data.flow_data,
                                        flow_scratch_data.scratch_data.get_min_cell_size());

                                    flux_liquid += viscous_interface_flux_liquid;
                                    flux_gas += viscous_interface_flux_gas;
                                  }
                              }

                            eval_point_interface_liquid.submit_value(-flux_liquid, q);
                            eval_point_interface_gas.submit_value(-flux_gas, q);

                            if ((is_viscous_liquid or is_viscous_gas) and
                                flow_scratch_data.flow_data.interface_jump_conditions.type ==
                                  InterfaceNumericalMethod::HLLP0_and_SIPG)
                              {
                                const auto [numerical_flux_gradient_liquid,
                                            numerical_flux_gradient_gas] =
                                  calculate_viscous_interface_flux_gradient<
                                    dim,
                                    number,
                                    ConservedVariablesType,
                                    ConservedVariablesGradType>(w_liquid,
                                                                w_gas,
                                                                normal,
                                                                visc_ave_weight_phase_1,
                                                                visc_ave_weight_phase_2,
                                                                viscous_terms_liquid,
                                                                viscous_terms_gas,
                                                                flow_scratch_data.flow_data);

                                eval_point_interface_liquid.submit_gradient(
                                  -numerical_flux_gradient_liquid, q);
                                eval_point_interface_gas.submit_gradient(
                                  -numerical_flux_gradient_gas, q);
                              }
                          }
                      }
                    else
                      Assert(false, dealii::ExcNotImplemented());

                    eval_point_interface_liquid.integrate(StridedArrayView<number, n_lanes>(
                                  &eval_liquid_intersected.begin_dof_values()[0][lane],
                                  n_dofs_per_cell),
                                  EvaluationFlags::values |
                                  (flow_scratch_data.flow_data.interface_jump_conditions.type
                                    == InterfaceNumericalMethod::HLLP0_and_SIPG ?
                                    EvaluationFlags::gradients: EvaluationFlags::nothing) , true
                                  /*specify flag 'true' for summing the integrated values
                                   *into the solution values*/);

                    eval_point_interface_gas.integrate(StridedArrayView<number, n_lanes>(
                                  &eval_gas_intersected.begin_dof_values()[0][lane],
                                  n_dofs_per_cell),
                                  EvaluationFlags::values |
                                  (flow_scratch_data.flow_data.interface_jump_conditions.type
                                    == InterfaceNumericalMethod::HLLP0_and_SIPG ?
                                    EvaluationFlags::gradients : EvaluationFlags::nothing), true
                                  /*specify flag 'true' for summing the integrated values
                                   *into the solution values*/);
                  }
                eval_liquid_intersected.distribute_local_to_global(dst);
                eval_gas_intersected.distribute_local_to_global(dst);
              }
            break;
          }
        default:
          break;
      }
  }

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_face_rhs(const dealii::MatrixFree<dim, number> &,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const
  {
    const auto face_category = flow_scratch_data.scratch_data.get_face_range_category(face_range);
    const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);

    auto process_bulk_face_range = [&]<bool is_viscous>(auto &eval_m,
                                                        auto &eval_p,
                                                        auto &convective_terms,
                                                        auto &viscous_terms) {
      const auto eval_flags = EvaluationFlags::values |
                              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing);
      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m.reinit(face);
          eval_p.reinit(face);

          eval_m.gather_evaluate(src, eval_flags);
          eval_p.gather_evaluate(src, eval_flags);

          const auto interior_penalty_parameter =
            is_viscous ?
              0.5 * std::max(eval_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                             eval_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
              0.;

          for (const unsigned int q : eval_m.quadrature_point_indices())
            {
              const auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
                MeltPoolDG::Flow::rhs_face_integral_kernel<dim,
                                                           number,
                                                           FEFaceIntegrator<dim, dim + 2, number>,
                                                           is_viscous>(
                  eval_m, eval_p, q, interior_penalty_parameter, convective_terms, viscous_terms);

              eval_m.submit_value(flux_m, q);
              eval_p.submit_value(flux_p, q);
              if (is_viscous)
                {
                  eval_m.submit_gradient(grad_flux_m, q);
                  eval_p.submit_gradient(grad_flux_p, q);
                }
            }
          eval_m.integrate_scatter(eval_flags, dst);
          eval_p.integrate_scatter(eval_flags, dst);
        }
    };

    auto process_intersected_face_range = [&]<bool is_viscous>(auto              &eval_m_int,
                                                               auto              &eval_p_int,
                                                               const unsigned int mapping_idx,
                                                               const auto        &convective_terms,
                                                               const auto        &viscous_terms) {
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_m(
        *mapping_info_faces[mapping_idx], fe_point_temp);
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_p(
        *mapping_info_faces[mapping_idx], fe_point_temp);

      const auto eval_flags = EvaluationFlags::values |
                              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m_int.reinit(face);
          eval_m_int.read_dof_values(src);
          eval_p_int.reinit(face);
          eval_p_int.read_dof_values(src);

          eval_m_int.project_to_face(eval_flags);
          eval_p_int.project_to_face(eval_flags);

          const auto face_info =
            flow_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

          for (unsigned int lane = 0;
               lane <
               flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_face_batch(
                 face);
               ++lane)
            {
              eval_point_m.reinit(face_info.cells_interior[lane],
                                  static_cast<int>(face_info.interior_face_no));
              eval_point_p.reinit(face_info.cells_exterior[lane],
                                  static_cast<int>(face_info.exterior_face_no));

              eval_point_m.evaluate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                            eval_flags);

              eval_point_p.evaluate_in_face(&eval_p_int.get_scratch_data().begin()[0][lane],
                                            eval_flags);

              // factor 0.5 for interior face
              const dealii::VectorizedArray<number> interior_penalty_parameter =
                is_viscous ?
                  0.5 * std::max(
                          eval_m_int.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                          eval_p_int.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
                  0.;

              for (const unsigned int q : eval_point_m.quadrature_point_indices())
                {
                  const auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
                    MeltPoolDG::Flow::rhs_face_integral_kernel<
                      dim,
                      number,
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>,
                      is_viscous>(eval_point_m,
                                  eval_point_p,
                                  q,
                                  interior_penalty_parameter,
                                  convective_terms,
                                  viscous_terms);

                  eval_point_m.submit_value(flux_m, q);
                  eval_point_p.submit_value(flux_p, q);
                  if (is_viscous)
                    {
                      eval_point_m.submit_gradient(grad_flux_m, q);
                      eval_point_p.submit_gradient(grad_flux_p, q);
                    }
                }

              eval_point_m.integrate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                             eval_flags);

              eval_point_p.integrate_in_face(&eval_p_int.get_scratch_data().begin()[0][lane],
                                             eval_flags);
            }

          eval_m_int.collect_from_face(eval_flags, eval_m_int.begin_dof_values());
          eval_p_int.collect_from_face(eval_flags, eval_p_int.begin_dof_values());

          eval_m_int.distribute_local_to_global(dst);
          eval_p_int.distribute_local_to_global(dst);
        }
    };

    switch (face_type)
      {
          case CutUtil::FaceType::inside_face_liquid: {
            auto [eval_liquid_m, eval_liquid_p] =
              create_face_integrators(CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(eval_liquid_m,
                                                                           eval_liquid_p,
                                                                           convective_terms_liquid,
                                                                           viscous_terms_liquid);
          }
          break;

          case CutUtil::FaceType::mixed_face_liquid_intersected: {
            auto eval_liquid_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, 0);
            auto eval_liquid_m = create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m,
              eval_liquid_p_intersected,
              convective_terms_liquid,
              viscous_terms_liquid);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_liquid: {
            auto eval_liquid_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
            auto eval_liquid_p = create_face_integrator(false, CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m_intersected,
              eval_liquid_p,
              convective_terms_liquid,
              viscous_terms_liquid);
          }
          break;

          case CutUtil::FaceType::inside_face_gas: {
            auto [eval_gas_m, eval_gas_p] =
              create_face_integrators(CutUtil::CellCategory::gas, dim + 2);
            process_bulk_face_range.template operator()<is_viscous_gas>(eval_gas_m,
                                                                        eval_gas_p,
                                                                        convective_terms_gas,
                                                                        viscous_terms_gas);
          }
          break;

          case CutUtil::FaceType::mixed_face_gas_intersected: {
            auto eval_gas_m = create_face_integrator(true, CutUtil::CellCategory::gas, dim + 2);
            auto eval_gas_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, dim + 2);
            process_bulk_face_range.template operator()<is_viscous_gas>(eval_gas_m,
                                                                        eval_gas_p_intersected,
                                                                        convective_terms_gas,
                                                                        viscous_terms_gas);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_gas: {
            auto eval_gas_p = create_face_integrator(false, CutUtil::CellCategory::gas, dim + 2);
            auto eval_gas_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, dim + 2);
            process_bulk_face_range.template operator()<is_viscous_gas>(eval_gas_m_intersected,
                                                                        eval_gas_p,
                                                                        convective_terms_gas,
                                                                        viscous_terms_gas);
          }
          break;

          case CutUtil::FaceType::intersected_face: {
            auto [eval_liquid_m_intersected, eval_liquid_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, 0);
            auto [eval_gas_m_intersected, eval_gas_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, dim + 2);
            process_intersected_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m_intersected,
              eval_liquid_p_intersected,
              0,
              convective_terms_liquid,
              viscous_terms_liquid);
            process_intersected_face_range.template operator()<is_viscous_gas>(
              eval_gas_m_intersected,
              eval_gas_p_intersected,
              1,
              convective_terms_gas,
              viscous_terms_gas);
            break;
          }
        default:
          break;
      }
  }

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_boundary_face_rhs(const dealii::MatrixFree<dim, number> &,
                                  VectorType                          &dst,
                                  const VectorType                    &src,
                                  const std::pair<unsigned, unsigned> &face_range) const
  {
    const auto face_category = flow_scratch_data.scratch_data.get_face_range_category(face_range);

    auto process_bulk_face_range =
      [&]<bool is_viscous>(auto &eval_m, auto &convective_terms, auto &viscous_terms) {
        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            eval_m.reinit(face);
            eval_m.gather_evaluate(src,
                                   dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients);

            const dealii::VectorizedArray<number> interior_penalty_parameter =
              is_viscous ? eval_m.read_cell_data(flow_scratch_data.interior_penalty_parameter) : 0.;

            for (const unsigned int q : eval_m.quadrature_point_indices())
              {
                const auto [flux_m, grad_flux_m] =
                  Flow::rhs_boundary_face_integral_kernel<dim,
                                                          number,
                                                          FEFaceIntegrator<dim, dim + 2, number>,
                                                          is_viscous>(
                    eval_m,
                    q,
                    flow_scratch_data.scratch_data.get_matrix_free().get_boundary_id(face),
                    interior_penalty_parameter,
                    convective_terms,
                    viscous_terms,
                    flow_scratch_data);
                eval_m.submit_value(flux_m, q);
                if (is_viscous)
                  eval_m.submit_gradient(grad_flux_m, q);
              }

            eval_m.integrate_scatter(EvaluationFlags::values |
                                       (is_viscous ? EvaluationFlags::gradients :
                                                     EvaluationFlags::nothing),
                                     dst);
          }
      };

    auto process_intersected_face_range = [&]<bool is_viscous>(auto              &eval_m_int,
                                                               const unsigned int mapping_idx,
                                                               auto              &convective_terms,
                                                               auto              &viscous_terms) {
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_m_int(
        *mapping_info_faces[mapping_idx], fe_point_temp);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m_int.reinit(face);
          eval_m_int.read_dof_values(src);

          eval_m_int.project_to_face(EvaluationFlags::values | EvaluationFlags::gradients);

          const auto face_info =
            flow_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

          for (unsigned int lane = 0;
               lane <
               flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_face_batch(
                 face);
               ++lane)
            {
              eval_point_m_int.reinit(face_info.cells_interior[lane],
                                      static_cast<int>(face_info.interior_face_no));

              eval_point_m_int.evaluate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                                EvaluationFlags::values |
                                                  EvaluationFlags::gradients);

              const dealii::VectorizedArray<number> interior_penalty_parameter =
                is_viscous ?
                  eval_m_int.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                  0.;

              for (const unsigned int q : eval_point_m_int.quadrature_point_indices())
                {
                  const auto [flux_m, grad_flux_m] =
                    MeltPoolDG::Flow::rhs_boundary_face_integral_kernel<
                      dim,
                      number,
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>,
                      is_viscous>(eval_point_m_int,
                                  q,
                                  flow_scratch_data.scratch_data.get_matrix_free().get_boundary_id(
                                    face),
                                  interior_penalty_parameter,
                                  convective_terms,
                                  viscous_terms,
                                  flow_scratch_data);

                  eval_point_m_int.submit_value(flux_m, q);
                  if (is_viscous)
                    eval_point_m_int.submit_gradient(grad_flux_m, q);
                }

              eval_point_m_int.integrate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                                 dealii::EvaluationFlags::values |
                                                   dealii::EvaluationFlags::gradients);
            }

          eval_m_int.collect_from_face(dealii::EvaluationFlags::values |
                                         dealii::EvaluationFlags::gradients,
                                       eval_m_int.begin_dof_values());

          eval_m_int.distribute_local_to_global(dst);
        }
    };

    switch (face_category.first)
      {
          case CutUtil::CellCategory::liquid: {
            auto eval_liquid_m = create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(eval_liquid_m,
                                                                           convective_terms_liquid,
                                                                           viscous_terms_liquid);
            break;
          }
          case CutUtil::CellCategory::gas: {
            auto eval_gas_m = create_face_integrator(true, CutUtil::CellCategory::gas, dim + 2);
            process_bulk_face_range.template operator()<is_viscous_gas>(eval_gas_m,
                                                                        convective_terms_gas,
                                                                        viscous_terms_gas);
            break;
          }
          case CutUtil::CellCategory::intersected: {
            auto eval_liquid_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
            auto eval_gas_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, dim + 2);
            process_intersected_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m_intersected, 0, convective_terms_liquid, viscous_terms_liquid);
            process_intersected_face_range.template operator()<is_viscous_gas>(
              eval_gas_m_intersected, 1, convective_terms_gas, viscous_terms_gas);
            break;
          }
        default:
          break;
      }
  }

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_cell_lhs(const dealii::MatrixFree<dim, number> &,
                         VectorType                          &dst,
                         const VectorType                    &src,
                         const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto cell_category = flow_scratch_data.scratch_data.get_cell_range_category(cell_range);

    // Processing function for non-intersected cells
    auto process_bulk_cell_range = [&](auto &eval) {
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval.reinit(cell);
          eval.gather_evaluate(src, dealii::EvaluationFlags::values);

          for (const unsigned int q : eval.quadrature_point_indices())
            eval.submit_value(eval.get_value(q), q);

          eval.integrate_scatter(dealii::EvaluationFlags::values, dst);
        }
    };

    // Processing function for intersected cells
    auto process_intersected_cell_range = [&](auto &eval_intersected, auto &eval_point) {
      constexpr unsigned int n_lanes = VectorizedArray<number>::size();

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval_intersected.reinit(cell);
          eval_intersected.read_dof_values(src);

          for (unsigned int lane = 0;
               lane <
               flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                 cell);
               ++lane)
            {
              eval_point.reinit(cell * n_lanes + lane);
              eval_point.evaluate(StridedArrayView<const number, n_lanes>(
                                    &eval_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                                  dealii::EvaluationFlags::values);

              for (const unsigned int q : eval_point.quadrature_point_indices())
                eval_point.submit_value(eval_point.get_value(q), q);

              eval_point.integrate(
                StridedArrayView<number, n_lanes>(&eval_intersected.begin_dof_values()[0][lane],
                                                  n_dofs_per_cell),
                dealii::EvaluationFlags::values);
            }
          eval_intersected.distribute_local_to_global(dst);
        }
    };

    switch (cell_category)
      {
          case CutUtil::CellCategory::liquid: {
            auto eval_liquid = create_cell_integrator(CutUtil::CellCategory::liquid, 0);
            process_bulk_cell_range(eval_liquid);
          }
          break;

          case CutUtil::CellCategory::gas: {
            auto eval_gas = create_cell_integrator(CutUtil::CellCategory::gas, dim + 2);
            process_bulk_cell_range(eval_gas);
          }
          break;

          case CutUtil::CellCategory::intersected: {
            auto eval_liquid_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, 0);
            auto eval_gas_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, dim + 2);

            FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_liquid(
              *mapping_info_cells[0], fe_point_temp);
            FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_gas(
              *mapping_info_cells[1], fe_point_temp);

            process_intersected_cell_range(eval_liquid_intersected, eval_point_liquid);
            process_intersected_cell_range(eval_gas_intersected, eval_point_gas);
          }
          break;

        default:
          break;
      }
  }

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_face_lhs(const dealii::MatrixFree<dim, number> &,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const
  {
    const auto face_category = flow_scratch_data.scratch_data.get_face_range_category(face_range);
    const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);

    const number cell_side_length       = flow_scratch_data.scratch_data.get_min_cell_size();
    const number cell_side_length_pow_3 = Utilities::fixed_power<3>(cell_side_length);
    const number cell_side_length_pow_5 = (flow_scratch_data.flow_data.fe.degree == 2) ?
                                            Utilities::fixed_power<5>(cell_side_length) :
                                            0.;

    auto apply_ghost_penalty = [&](auto &eval_m, auto &eval_p) {
      EvaluationFlags::EvaluationFlags evaluation_flags =
        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients |
        ((flow_scratch_data.flow_data.fe.degree == 2) ? dealii::EvaluationFlags::hessians :
                                                        dealii::EvaluationFlags::nothing);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m.reinit(face);
          eval_m.gather_evaluate(src, evaluation_flags);

          eval_p.reinit(face);
          eval_p.gather_evaluate(src, evaluation_flags);

          for (const unsigned int q : eval_m.quadrature_point_indices())
            {
              const auto w_minus = eval_m.get_value(q);
              const auto w_plus  = eval_p.get_value(q);

              const auto w_normal_grad_minus = eval_m.get_normal_derivative(q);
              const auto w_normal_grad_plus  = eval_p.get_normal_derivative(q);

              const auto ghost_penalty_term_0 =
                (w_minus - w_plus) *
                flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_0 *
                cell_side_length;

              const auto ghost_penalty_term_1 =
                (w_normal_grad_minus - w_normal_grad_plus) *
                flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_1 *
                cell_side_length_pow_3;

              if (flow_scratch_data.flow_data.fe.degree == 2)
                {
                  const auto w_normal_hessian_minus = eval_m.get_normal_hessian(q);
                  const auto w_normal_hessian_plus  = eval_p.get_normal_hessian(q);

                  const auto ghost_penalty_term_2 =
                    (w_normal_hessian_minus - w_normal_hessian_plus) *
                    flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_2 *
                    cell_side_length_pow_5;

                  eval_m.submit_normal_hessian(ghost_penalty_term_2, q);
                  eval_p.submit_normal_hessian(-ghost_penalty_term_2, q);
                }
              eval_m.submit_normal_derivative(ghost_penalty_term_1, q);
              eval_p.submit_normal_derivative(-ghost_penalty_term_1, q);

              eval_m.submit_value(ghost_penalty_term_0, q);
              eval_p.submit_value(-ghost_penalty_term_0, q);
            }
          eval_m.integrate_scatter(evaluation_flags, dst);
          eval_p.integrate_scatter(evaluation_flags, dst);
        }
    };

    switch (face_type)
      {
          case CutUtil::FaceType::mixed_face_liquid_intersected: {
            auto eval_liquid_m = create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
            auto eval_liquid_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, 0);
            apply_ghost_penalty(eval_liquid_m, eval_liquid_p_intersected);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_liquid: {
            auto eval_liquid_p = create_face_integrator(false, CutUtil::CellCategory::liquid, 0);
            auto eval_liquid_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
            apply_ghost_penalty(eval_liquid_m_intersected, eval_liquid_p);
          }
          break;

          case CutUtil::FaceType::mixed_face_gas_intersected: {
            auto eval_gas_m = create_face_integrator(true, CutUtil::CellCategory::gas, dim + 2);
            auto eval_gas_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, dim + 2);
            apply_ghost_penalty(eval_gas_m, eval_gas_p_intersected);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_gas: {
            auto eval_gas_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, dim + 2);
            auto eval_gas_p = create_face_integrator(false, CutUtil::CellCategory::gas, dim + 2);
            apply_ghost_penalty(eval_gas_m_intersected, eval_gas_p);
          }
          break;

          case CutUtil::FaceType::intersected_face: {
            auto [eval_liquid_m_intersected, eval_liquid_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, 0);
            auto [eval_gas_m_intersected, eval_gas_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, dim + 2);
            apply_ghost_penalty(eval_liquid_m_intersected, eval_liquid_p_intersected);
            apply_ghost_penalty(eval_gas_m_intersected, eval_gas_p_intersected);
          }
          break;

        default:
          break;
      }
  }

  template <unsigned int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_boundary_face_lhs(const dealii::MatrixFree<dim, number> &,
                                  VectorType &,
                                  const VectorType &,
                                  const std::pair<unsigned, unsigned> &) const
  {
    // nothing to do here
  }

  template class CompressibleMultiphaseOperator<1, double, true, true>;
  template class CompressibleMultiphaseOperator<2, double, true, true>;
  template class CompressibleMultiphaseOperator<3, double, true, true>;
  template class CompressibleMultiphaseOperator<1, double, true, false>;
  template class CompressibleMultiphaseOperator<2, double, true, false>;
  template class CompressibleMultiphaseOperator<3, double, true, false>;
  template class CompressibleMultiphaseOperator<1, double, false, true>;
  template class CompressibleMultiphaseOperator<2, double, false, true>;
  template class CompressibleMultiphaseOperator<3, double, false, true>;
  template class CompressibleMultiphaseOperator<1, double, false, false>;
  template class CompressibleMultiphaseOperator<2, double, false, false>;
  template class CompressibleMultiphaseOperator<3, double, false, false>;
} // namespace MeltPoolDG::Multiphase
