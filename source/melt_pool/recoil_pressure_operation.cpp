#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>

namespace MeltPoolDG::MeltPool
{
  template <int dim>
  RecoilPressureOperation<dim>::RecoilPressureOperation(const ScratchData<dim> &  scratch_data_in,
                                                        const Parameters<double> &data_in,
                                                        const unsigned int flow_vel_dof_idx_in,
                                                        const unsigned int flow_vel_quad_idx_in,
                                                        const unsigned int flow_pressure_dof_idx_in,
                                                        const unsigned int ls_dof_idx_in,
                                                        const unsigned int temp_dof_idx_in)
    : scratch_data(scratch_data_in)
    , recoil_pressure_data(data_in.recoil)
    , boiling_temperature(data_in.material.boiling_temperature)
    , flow_vel_dof_idx(flow_vel_dof_idx_in)
    , flow_vel_quad_idx(flow_vel_quad_idx_in)
    , flow_pressure_dof_idx(flow_pressure_dof_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , temp_dof_idx(temp_dof_idx_in)
    , do_level_set_pressure_gradient_interpolation(scratch_data.is_FE_Q_iso_Q_1(ls_dof_idx_in))
  {
    AssertThrow(boiling_temperature > 0.0,
                ExcMessage("The boiling temperature must be greater than zero! Abort..."));

    if (do_level_set_pressure_gradient_interpolation)
      {
        ls_to_pressure_grad_interpolation_matrix =
          UtilityFunctions::create_dof_interpolation_matrix<dim>(
            scratch_data.get_dof_handler(flow_pressure_dof_idx),
            scratch_data.get_dof_handler(ls_dof_idx),
            true /*do_matrix_free*/);
      }

    if (recoil_pressure_data.delta_function_type ==
        DiracDeltaFunctionApproximationType::phase_weighted_delta)
      delta_phase_weighted = std::make_unique<DeltaApproximationPhaseWeighted<double>>(
        recoil_pressure_data.delta_approximation_phase_weighted);
    else if (recoil_pressure_data.delta_function_type ==
             DiracDeltaFunctionApproximationType::quad_phase_weighted_delta)
      delta_phase_weighted = std::make_unique<DeltaApproximationQuadPhaseWeighted<double>>(
        recoil_pressure_data.delta_approximation_phase_weighted);
  }

  template <int dim>
  void
  RecoilPressureOperation<dim>::compute_recoil_pressure_force(
    VectorType &      force_rhs,
    const VectorType &level_set_as_heaviside,
    const VectorType &temperature,
    bool              zero_out) const
  {
    temperature.update_ghost_values();
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto &      force_rhs,
          const auto &level_set_as_heaviside,
          auto        macro_cells) {
        FECellIntegrator<dim, 1, double> level_set(matrix_free, ls_dof_idx, flow_vel_quad_idx);

        FECellIntegrator<dim, dim, double> recoil_pressure(matrix_free,
                                                           flow_vel_dof_idx,
                                                           flow_vel_quad_idx);

        FECellIntegrator<dim, 1, double> temperature_val(matrix_free,
                                                         temp_dof_idx,
                                                         flow_vel_quad_idx);

        FECellIntegrator<dim, 1, double> interpolated_level_set_to_pressure_space(
          matrix_free, flow_pressure_dof_idx, flow_vel_quad_idx);

        auto &used_level_set = do_level_set_pressure_gradient_interpolation ?
                                 interpolated_level_set_to_pressure_space :
                                 level_set;

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

            if (delta_phase_weighted)
              used_level_set.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            else
              used_level_set.evaluate(EvaluationFlags::gradients);

            temperature_val.reinit(cell);
            temperature_val.read_dof_values_plain(temperature);
            temperature_val.evaluate(EvaluationFlags::values);

            recoil_pressure.reinit(cell);

            for (unsigned int q_index = 0; q_index < recoil_pressure.n_q_points; ++q_index)
              {
                const auto &t = temperature_val.get_value(q_index);

                VectorizedArray<double> recoil_pressure_coefficient = 0;

                for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
                  recoil_pressure_coefficient[v] = compute_recoil_pressure_coefficient(t[v]);

                VectorizedArray<double> weight(1.0);

                if (delta_phase_weighted)
                  weight = delta_phase_weighted->compute_weight(used_level_set.get_value(q_index));

                recoil_pressure.submit_value(recoil_pressure_coefficient *
                                               used_level_set.get_gradient(q_index) * weight,
                                             q_index);
              }
            recoil_pressure.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      level_set_as_heaviside,
      zero_out);
    temperature.zero_out_ghost_values();
  }

  template <int dim>
  inline double
  RecoilPressureOperation<dim>::compute_recoil_pressure_coefficient(const double T) const
  {
    return compute_recoil_pressure_coefficient(T,
                                               recoil_pressure_data.pressure_constant,
                                               recoil_pressure_data.temperature_constant,
                                               boiling_temperature);
  }

  template <int dim>
  double
  RecoilPressureOperation<dim>::compute_recoil_pressure_coefficient(
    const double  T,
    const double &pressure_constant,
    const double &temperature_constant,
    const double &boiling_temperature)
  {
    return pressure_constant *
           std::exp(-temperature_constant * (1. / T - 1. / boiling_temperature));
  }

  template class RecoilPressureOperation<1>;
  template class RecoilPressureOperation<2>;
  template class RecoilPressureOperation<3>;
} // namespace MeltPoolDG::MeltPool
