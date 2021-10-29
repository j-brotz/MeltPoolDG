#include <meltpooldg/flow/surface_tension_operation.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim>
  SurfaceTensionOperation<dim>::SurfaceTensionOperation(
    const SurfaceTensionData<double> &data_in,
    const ScratchData<dim> &          scratch_data,
    const VectorType &                level_set_as_heaviside,
    const VectorType &                solution_curvature,
    const unsigned int                ls_dof_idx,
    const unsigned int                curv_dof_idx,
    const unsigned int                flow_vel_dof_idx,
    const unsigned int                flow_pressure_hanging_nodes_dof_idx_in,
    const unsigned int                flow_vel_quad_idx)
    : data(data_in)
    , scratch_data(scratch_data)
    , level_set_as_heaviside(level_set_as_heaviside)
    , solution_curvature(solution_curvature)
    , ls_dof_idx(ls_dof_idx)
    , curv_dof_idx(curv_dof_idx)
    , flow_vel_dof_idx(flow_vel_dof_idx)
    , flow_pressure_hanging_nodes_dof_idx(flow_pressure_hanging_nodes_dof_idx_in)
    , flow_vel_quad_idx(flow_vel_quad_idx)
    , do_level_set_pressure_gradient_interpolation(scratch_data.is_FE_Q_iso_Q_1(ls_dof_idx))
  {
    if (do_level_set_pressure_gradient_interpolation)
      {
        ls_to_pressure_grad_interpolation_matrix =
          UtilityFunctions::create_dof_interpolation_matrix<dim>(
            scratch_data.get_dof_handler(flow_pressure_hanging_nodes_dof_idx),
            scratch_data.get_dof_handler(ls_dof_idx),
            true);
      }

    if (data.delta_function_type == DiracDeltaFunctionApproximationType::phase_weighted_delta)
      {
        delta_phase_weighted = std::make_unique<DeltaApproximationPhaseWeighted<double>>(
          data.delta_approximation_phase_weighted);
      }

    //@todo add assert for parameters
  }

  template <int dim>
  void
  SurfaceTensionOperation<dim>::reinit(const unsigned int temp_dof_idx_in,
                                       const unsigned int normal_dof_idx_in,
                                       VectorType *       temperature_in,
                                       BlockVectorType *  solution_normal_vector_in)
  {
    temp_dof_idx           = temp_dof_idx_in;
    normal_dof_idx         = normal_dof_idx_in;
    temperature            = temperature_in;
    solution_normal_vector = solution_normal_vector_in;
  }

  template <int dim>
  void
  SurfaceTensionOperation<dim>::compute_surface_tension(VectorType &force_rhs, const bool zero_out)
  {
    solution_curvature.update_ghost_values();

    if (temperature)
      {
        solution_normal_vector->update_ghost_values();
        temperature->update_ghost_values();
      }

    const double tolerance_normal_vector =
      UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                            scratch_data.get_mapping());

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto &      force_rhs,
          const auto &level_set_as_heaviside,
          auto        macro_cells) {
        FECellIntegrator<dim, 1, double> level_set(matrix_free, ls_dof_idx, flow_vel_quad_idx);

        FECellIntegrator<dim, 1, double> curvature(matrix_free, curv_dof_idx, flow_vel_quad_idx);

        FECellIntegrator<dim, 1, double> interpolated_level_set_to_pressure_space(
          matrix_free, flow_pressure_hanging_nodes_dof_idx, flow_vel_quad_idx);

        std::unique_ptr<FECellIntegrator<dim, dim, double>> normal_vec;
        std::unique_ptr<FECellIntegrator<dim, 1, double>>   temperature_val;

        auto &used_level_set = do_level_set_pressure_gradient_interpolation ?
                                 interpolated_level_set_to_pressure_space :
                                 level_set;

        if (temperature)
          {
            normal_vec      = std::make_unique<FECellIntegrator<dim, dim, double>>(matrix_free,
                                                                              normal_dof_idx,
                                                                              flow_vel_quad_idx);
            temperature_val = std::make_unique<FECellIntegrator<dim, 1, double>>(matrix_free,
                                                                                 temp_dof_idx,
                                                                                 flow_vel_quad_idx);
          }

        FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                           flow_vel_dof_idx,
                                                           flow_vel_quad_idx);

        const double &alpha0         = data.surface_tension_coefficient;
        const double &d_alpha0       = data.temperature_dependent_surface_tension_coefficient;
        const double  alpha_residual = alpha0 * data.coefficient_residual_fraction;
        const auto    T0             = VectorizedArray<double>(data.reference_temperature);

        auto alpha = VectorizedArray<double>(alpha0);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            level_set.reinit(cell);
            level_set.read_dof_values_plain(level_set_as_heaviside);

            if (do_level_set_pressure_gradient_interpolation)
              {
                interpolated_level_set_to_pressure_space.reinit(cell);

                UtilityFunctions::compute_gradient_at_interpolated_dof_values<dim>(
                  level_set,
                  interpolated_level_set_to_pressure_space,
                  ls_to_pressure_grad_interpolation_matrix);
              }

            if (data.delta_function_type ==
                DiracDeltaFunctionApproximationType::norm_of_indicator_gradient)
              used_level_set.evaluate(EvaluationFlags::gradients);
            else if (data.delta_function_type ==
                     DiracDeltaFunctionApproximationType::phase_weighted_delta)
              used_level_set.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            else
              AssertThrow(false, ExcNotImplemented());

            surface_tension.reinit(cell);

            curvature.reinit(cell);
            curvature.read_dof_values_plain(solution_curvature);
            curvature.evaluate(EvaluationFlags::values);

            if (temperature)
              {
                normal_vec->reinit(cell);
                normal_vec->read_dof_values_plain(*solution_normal_vector);
                normal_vec->evaluate(EvaluationFlags::values);

                temperature_val->reinit(cell);
                temperature_val->read_dof_values_plain(*temperature);
                temperature_val->evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
              }

            for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
              {
                auto delta = used_level_set.get_gradient(q_index).norm();
                if (data.delta_function_type ==
                    DiracDeltaFunctionApproximationType::phase_weighted_delta)
                  delta *= delta_phase_weighted->compute_weight(used_level_set.get_value(q_index));

                if (temperature)
                  {
                    const auto n =
                      MeltPoolDG::VectorTools::normalize<dim>(normal_vec->get_value(q_index),
                                                              tolerance_normal_vector);
                    const auto T      = temperature_val->get_value(q_index);
                    const auto grad_T = temperature_val->get_gradient(q_index);

                    Tensor<1, dim, VectorizedArray<double>> temp_surf_ten;

                    for (unsigned int i = 0; i < dim; ++i)
                      for (unsigned int j = 0; j < dim; ++j)
                        temp_surf_ten[i] = (i == j) ?
                                             (make_vectorized_array<double>(1.) - n[i] * n[j]) *
                                               (-d_alpha0) * grad_T[j] :
                                             -(n[i] * n[j]) * (-d_alpha0) * grad_T[j];

                    alpha = VectorizedArray<double>(alpha0) -
                            VectorizedArray<double>(d_alpha0) * (T - T0);

                    // The surface tension must not become negative or smaller than its residual
                    // value.
                    alpha = compare_and_apply_mask<SIMDComparison::less_than>(alpha,
                                                                              alpha_residual,
                                                                              alpha_residual,
                                                                              alpha);
                    surface_tension.submit_value(alpha * n * curvature.get_value(q_index) * delta +
                                                   temp_surf_ten * delta,
                                                 q_index);
                  }
                else
                  surface_tension.submit_value(alpha * used_level_set.get_gradient(q_index) *
                                                 curvature.get_value(q_index),
                                               q_index);
              }
            surface_tension.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      level_set_as_heaviside,
      zero_out);

    solution_curvature.zero_out_ghost_values();

    if (temperature)
      {
        temperature->zero_out_ghost_values();
        solution_normal_vector->zero_out_ghost_values();
      }
  }

  template class SurfaceTensionOperation<1>;
  template class SurfaceTensionOperation<2>;
  template class SurfaceTensionOperation<3>;
} // namespace MeltPoolDG::Flow
