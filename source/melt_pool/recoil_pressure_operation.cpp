#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>

namespace MeltPoolDG::MeltPool
{
  template <typename number>
  RecoilPressurePhenomenologicalModel<number>::RecoilPressurePhenomenologicalModel(
    const RecoilPressureData<number> &recoil_data,
    const number                      boiling_temperature)
    : recoil_data(recoil_data)
    , boiling_temperature(boiling_temperature)
  {
    AssertThrow(boiling_temperature > 0.0,
                ExcMessage("The boiling temperature must be greater than zero! Abort..."));

    AssertThrow(recoil_data.activation_temperature <= boiling_temperature,
                ExcMessage(
                  "The activation temperature for the recoil pressure must be smaller than "
                  " or equal to the boiling temperature. Abort..."));
  }

  template <typename number>
  number
  RecoilPressurePhenomenologicalModel<number>::compute_recoil_pressure_coefficient(
    const number T) const
  {
    const number T_ac = recoil_data.activation_temperature;

    if (T < T_ac)
      return 0.0;

    // In order to smoothly activate the recoil pressure, a scaling coefficient is computed and
    // multiplied with the recoil pressure coefficient
    //
    //                  -
    //                 |     0         if T <= T_ac
    //                 |
    //                 |  T - T_ac
    // scaling_coeff = |  --------     if T_ac < T < T_v
    //                 |  T_v - T_ac
    //                 |
    //                 |     1         if T >= T_v
    //                  -
    //
    // with T_ac the activation temperature of the recoil pressure and T_v the boiling temperature.

    const number T_v = boiling_temperature;
    const number c_p = recoil_data.pressure_constant;
    const number c_T = recoil_data.temperature_constant;

    const number scaling_coeff =
      (T >= T_v || std::abs(T_v - T_ac) < 1e-10) ? 1. : (T - T_ac) / (T_v - T_ac);

    return scaling_coeff * c_p * std::exp(-c_T * (1. / T - 1. / T_v));
  }



  template <typename number>
  RecoilPressureHybridModel<number>::RecoilPressureHybridModel(
    const RecoilPressureData<number> &recoil_data,
    const MaterialData<number> &      material)
    : recoil_data(recoil_data)
    , boiling_temperature(material.boiling_temperature)
    , density_coeff(1. / material.first.density - 1. / material.second.density)
    , recoil_phenomenological(recoil_data, material.boiling_temperature)
  {}

  template <typename number>
  number
  RecoilPressureHybridModel<number>::compute_recoil_pressure_coefficient(const number T,
                                                                         const number m_dot) const
  {
    const number T_ac = recoil_data.activation_temperature;

    if (T < T_ac)
      return 0.0;

    // In order to smoothly activate the recoil pressure, a scaling coefficient is computed and
    // multiplied with the recoil pressure coefficient
    //
    //                  -
    //                 |     0         if T <= T_ac
    //                 |
    //                 |  T - T_ac
    // scaling_coeff = |  --------     if T_ac < T < T_v
    //                 |  T_v - T_ac
    //                 |
    //                 |     1         if T >= T_v
    //                  -
    //
    // with T_ac the activation temperature of the recoil pressure and T_v the boiling temperature.

    const number T_v = boiling_temperature;

    const number scaling_coeff =
      (T >= T_v || std::abs(T_v - T_ac) < 1e-10) ? 1. : (T - T_ac) / (T_v - T_ac);

    return recoil_phenomenological.compute_recoil_pressure_coefficient(T) -
           scaling_coeff * std::pow(m_dot, 2) * density_coeff;
  }



  template <int dim>
  RecoilPressureOperation<dim>::RecoilPressureOperation(const ScratchData<dim> &  scratch_data_in,
                                                        const Parameters<double> &data_in,
                                                        const unsigned int flow_vel_dof_idx_in,
                                                        const unsigned int flow_vel_quad_idx_in,
                                                        const unsigned int flow_pressure_dof_idx_in,
                                                        const unsigned int ls_dof_idx_in,
                                                        const unsigned int temp_dof_idx_in)
    : scratch_data(scratch_data_in)
    , flow_vel_dof_idx(flow_vel_dof_idx_in)
    , flow_vel_quad_idx(flow_vel_quad_idx_in)
    , flow_pressure_dof_idx(flow_pressure_dof_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , temp_dof_idx(temp_dof_idx_in)
    , do_level_set_pressure_gradient_interpolation(scratch_data.is_FE_Q_iso_Q_1(ls_dof_idx_in))
    , model_type(data_in.recoil.model_type)
    , delta_phase_weighted(create_phase_weighted_delta_approximation(
        data_in.recoil.delta_approximation_phase_weighted))
  {
    switch (data_in.recoil.model_type)
      {
        default:
        case RecoilPressureModelType::phenomenological:
          recoil_pressure_model =
            std::make_unique<const RecoilPressurePhenomenologicalModel<double>>(
              data_in.recoil, data_in.material.boiling_temperature);
          break;
        case RecoilPressureModelType::hybrid:
          recoil_pressure_model =
            std::make_unique<const RecoilPressureHybridModel<double>>(data_in.recoil,
                                                                      data_in.material);
          break;
      }

    if (do_level_set_pressure_gradient_interpolation)
      {
        ls_to_pressure_grad_interpolation_matrix =
          UtilityFunctions::create_dof_interpolation_matrix<dim>(
            scratch_data.get_dof_handler(flow_pressure_dof_idx),
            scratch_data.get_dof_handler(ls_dof_idx),
            true /*do_matrix_free*/);
      }
  }

  template <int dim>
  void
  RecoilPressureOperation<dim>::compute_recoil_pressure_force(
    VectorType &       force_rhs,
    const VectorType & level_set_as_heaviside,
    const VectorType & temperature,
    const VectorType & evaporative_mass_flux,
    const unsigned int evapor_dof_idx,
    bool               zero_out) const
  {
    temperature.update_ghost_values();
    if (model_type == RecoilPressureModelType::hybrid)
      evaporative_mass_flux.update_ghost_values();

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

        std::unique_ptr<FECellIntegrator<dim, 1, double>> evapor_flux_val;

        if (model_type == RecoilPressureModelType::hybrid)
          evapor_flux_val = std::make_unique<FECellIntegrator<dim, 1, double>>(matrix_free,
                                                                               evapor_dof_idx,
                                                                               flow_vel_quad_idx);

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

            if (evapor_flux_val)
              {
                evapor_flux_val->reinit(cell);
                evapor_flux_val->read_dof_values_plain(evaporative_mass_flux);
                evapor_flux_val->evaluate(EvaluationFlags::values);
              }

            recoil_pressure.reinit(cell);

            for (unsigned int q_index = 0; q_index < recoil_pressure.n_q_points; ++q_index)
              {
                const auto &t = temperature_val.get_value(q_index);

                VectorizedArray<double> m_dot = 0.0;
                if (evapor_flux_val)
                  m_dot = evapor_flux_val->get_value(q_index);

                VectorizedArray<double> recoil_pressure_coefficient = 0.0;

                for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
                  recoil_pressure_coefficient[v] =
                    evapor_flux_val ?
                      recoil_pressure_model->compute_recoil_pressure_coefficient(t[v], m_dot[v]) :
                      recoil_pressure_model->compute_recoil_pressure_coefficient(t[v]);

                VectorizedArray<double> weight = 1.0;

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
    if (model_type == RecoilPressureModelType::hybrid)
      evaporative_mass_flux.zero_out_ghost_values();
  }

  template class RecoilPressureOperation<1>;
  template class RecoilPressureOperation<2>;
  template class RecoilPressureOperation<3>;

  template class RecoilPressurePhenomenologicalModel<double>;
  template class RecoilPressureHybridModel<double>;
} // namespace MeltPoolDG::MeltPool
