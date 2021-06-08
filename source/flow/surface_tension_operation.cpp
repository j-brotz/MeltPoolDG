#include <meltpooldg/flow/surface_tension_operation.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim>
  void
  SurfaceTensionOperation<dim>::compute_surface_tension(
    VectorType &            force_rhs,
    const ScratchData<dim> &scratch_data,
    const VectorType &      level_set_as_heaviside,
    const VectorType &      curvature_vec,
    const double            surface_tension_coefficient,
    const unsigned int      ls_dof_idx,
    const unsigned int      curv_dof_idx,
    const unsigned int      flow_vel_hanging_nodes_dof_idx,
    const unsigned int      flow_quad_idx,
    const bool              zero_out)
  {
    curvature_vec.update_ghost_values();

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto &      force_rhs,
          const auto &level_set_as_heaviside,
          auto        macro_cells) {
        FECellIntegrator<dim, 1, double> level_set(matrix_free, ls_dof_idx, flow_quad_idx);

        FECellIntegrator<dim, 1, double> curvature(matrix_free, curv_dof_idx, flow_quad_idx);

        FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                           flow_vel_hanging_nodes_dof_idx,
                                                           flow_quad_idx);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            level_set.reinit(cell);
            level_set.read_dof_values_plain(level_set_as_heaviside);
            level_set.evaluate(EvaluationFlags::gradients);

            surface_tension.reinit(cell);

            curvature.reinit(cell);
            curvature.gather_evaluate(curvature_vec, EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
              {
                surface_tension.submit_value(surface_tension_coefficient *
                                               level_set.get_gradient(q_index) *
                                               curvature.get_value(q_index),
                                             q_index);
              }
            surface_tension.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      level_set_as_heaviside,
      zero_out);

    curvature_vec.zero_out_ghost_values();
  }

  template <int dim>
  void
  SurfaceTensionOperation<dim>::compute_temperature_dependent_surface_tension(
    const ScratchData<dim> &scratch_data,
    VectorType &            force_rhs,
    const VectorType &      level_set_as_heaviside,
    const VectorType &      solution_curvature,
    const VectorType &      temperature,
    const BlockVectorType & solution_normal_vector,
    const double            surface_tension_coefficient,
    const double            temperature_dependent_surface_tension_coefficient,
    const double            surface_tension_reference_temperature,
    const double            surface_tension_coefficient_residual_fraction,
    const unsigned int      ls_dof_idx,
    const unsigned int      curv_dof_idx,
    const unsigned int      normal_dof_idx,
    const unsigned int      flow_vel_dof_idx,
    const unsigned int      flow_vel_quad_idx,
    const unsigned int      temp_dof_idx,
    const bool              zero_out)
  {
    solution_curvature.update_ghost_values();
    temperature.update_ghost_values();
    solution_normal_vector.update_ghost_values();

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

        FECellIntegrator<dim, dim, double> normal_vec(matrix_free,
                                                      normal_dof_idx,
                                                      flow_vel_quad_idx);

        FECellIntegrator<dim, 1, double> temperature_val(matrix_free,
                                                         temp_dof_idx,
                                                         flow_vel_quad_idx);

        FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                           flow_vel_dof_idx,
                                                           flow_vel_quad_idx);

        const double &alpha0         = surface_tension_coefficient;
        const double &d_alpha0       = temperature_dependent_surface_tension_coefficient;
        const double  alpha_residual = alpha0 * surface_tension_coefficient_residual_fraction;
        const auto    T0 = VectorizedArray<double>(surface_tension_reference_temperature);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            level_set.reinit(cell);
            level_set.read_dof_values_plain(level_set_as_heaviside);
            level_set.evaluate(EvaluationFlags::gradients);

            surface_tension.reinit(cell);

            curvature.reinit(cell);
            curvature.read_dof_values_plain(solution_curvature);
            curvature.evaluate(EvaluationFlags::values);

            normal_vec.reinit(cell);
            normal_vec.read_dof_values_plain(solution_normal_vector);
            normal_vec.evaluate(EvaluationFlags::values);

            temperature_val.reinit(cell);
            temperature_val.read_dof_values_plain(temperature);
            temperature_val.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
              {
                const auto delta = level_set.get_gradient(q_index).norm();
                const auto n =
                  MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index),
                                                          tolerance_normal_vector);

                const auto T      = temperature_val.get_value(q_index);
                const auto grad_T = temperature_val.get_gradient(q_index);

                Tensor<1, dim, VectorizedArray<double>> temp_surf_ten;

                for (unsigned int i = 0; i < dim; ++i)
                  for (unsigned int j = 0; j < dim; ++j)
                    temp_surf_ten[i] = (i == j) ?
                                         (make_vectorized_array<double>(1.) - n[i] * n[j]) *
                                           (-d_alpha0) * grad_T[j] :
                                         -(n[i] * n[j]) * (-d_alpha0) * grad_T[j];

                auto alpha =
                  VectorizedArray<double>(alpha0) - VectorizedArray<double>(d_alpha0) * (T - T0);

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
            surface_tension.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      level_set_as_heaviside,
      zero_out);

    solution_curvature.zero_out_ghost_values();
    temperature.zero_out_ghost_values();
    solution_normal_vector.zero_out_ghost_values();
  }

  template class SurfaceTensionOperation<1>;
  template class SurfaceTensionOperation<2>;
  template class SurfaceTensionOperation<3>;
} // namespace MeltPoolDG::Flow
