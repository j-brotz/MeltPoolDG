#include <meltpooldg/phase_change/recoil_pressure_operation.templates.hpp>
//
#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/dof_tools.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <cmath>
#include <limits>
#include <vector>


namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /*******************************************************************
   * Phenomenological model
   *******************************************************************/
  template <typename number>
  RecoilPressurePhenomenologicalModel<number>::RecoilPressurePhenomenologicalModel(
    const RecoilPressureData<number> &recoil_data,
    const number                      boiling_temperature,
    const number                      molar_mass,
    const number                      latent_heat_evaporation)
    : recoil_data(recoil_data)
    , boiling_temperature(boiling_temperature)
    , molar_mass(molar_mass)
    , latent_heat_evaporation(latent_heat_evaporation)
  {
    AssertThrow(boiling_temperature > 0.0,
                ExcMessage("The boiling temperature must be greater than zero! Abort..."));

    AssertThrow(recoil_data.activation_temperature <= boiling_temperature,
                ExcMessage(
                  "The activation temperature for the recoil pressure must be smaller than "
                  " or equal to the boiling temperature. Abort..."));
  }

  /*******************************************************************
   * Hybrid model
   *******************************************************************/
  template <typename number>
  RecoilPressureHybridModel<number>::RecoilPressureHybridModel(
    const RecoilPressureData<number> &recoil_data,
    const MaterialData<number>       &material)
    : recoil_data(recoil_data)
    , boiling_temperature(material.boiling_temperature)
    , density_coeff(1. / material.gas.density - 1. / material.liquid.density)
    , recoil_phenomenological(recoil_data,
                              material.boiling_temperature,
                              material.molar_mass,
                              material.latent_heat_of_evaporation)
  {}

  template <int dim, typename number>
  RecoilPressureOperation<dim, number>::RecoilPressureOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const Parameters<number>            &data_in,
    const unsigned int                   flow_vel_dof_idx_in,
    const unsigned int                   flow_vel_quad_idx_in,
    const unsigned int                   flow_pressure_dof_idx_in,
    const unsigned int                   ls_dof_idx_in,
    const unsigned int                   heat_dof_idx_in)
    : scratch_data(scratch_data_in)
    , flow_vel_dof_idx(flow_vel_dof_idx_in)
    , flow_vel_quad_idx(flow_vel_quad_idx_in)
    , flow_pressure_dof_idx(flow_pressure_dof_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , heat_dof_idx(heat_dof_idx_in)
    , do_level_set_pressure_gradient_interpolation(scratch_data.is_FE_Q_iso_Q_1(ls_dof_idx_in))
    , model_type(data_in.evapor.recoil.type)
    , delta_phase_weighted(create_phase_weighted_delta_approximation(
        data_in.evapor.recoil.delta_approximation_phase_weighted))
    , dummy_temperature(std::nextafter(data_in.evapor.recoil.activation_temperature,
                                       -std::numeric_limits<number>::infinity()))
  {
    switch (data_in.evapor.recoil.type)
      {
        default:
        case RecoilPressureModelType::phenomenological:
          recoil_pressure_model =
            std::make_unique<const RecoilPressurePhenomenologicalModel<number>>(
              data_in.evapor.recoil,
              data_in.material.boiling_temperature,
              data_in.material.molar_mass,
              data_in.material.latent_heat_of_evaporation);
          break;
        case RecoilPressureModelType::hybrid:
          recoil_pressure_model =
            std::make_unique<const RecoilPressureHybridModel<number>>(data_in.evapor.recoil,
                                                                      data_in.material);
          break;
      }

    if (do_level_set_pressure_gradient_interpolation)
      {
        ls_to_pressure_grad_interpolation_matrix =
          DoFTools::create_dof_interpolation_matrix<dim, number>(
            scratch_data.get_dof_handler(flow_pressure_dof_idx),
            scratch_data.get_dof_handler(ls_dof_idx),
            true /*do_matrix_free*/);
      }
  }

  template <int dim, typename number>
  void
  RecoilPressureOperation<dim, number>::compute_recoil_pressure_force(
    VectorType        &force_rhs,
    const VectorType  &level_set_as_heaviside,
    const VectorType  &temperature,
    const VectorType  &evaporative_mass_flux,
    const unsigned int evapor_dof_idx,
    bool               zero_out) const
  {
    if (not temperature.has_ghost_elements())
      temperature.update_ghost_values();

    if (model_type == RecoilPressureModelType::hybrid and
        not evaporative_mass_flux.has_ghost_elements())
      evaporative_mass_flux.update_ghost_values();

    const CutUtil::CutPhaseType cut_type = scratch_data.get_cut_type(heat_dof_idx);

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto       &force_rhs,
          const auto &level_set_as_heaviside,
          auto        cell_range) {
        FECellIntegrator<dim, 1, number> heaviside_eval(matrix_free, ls_dof_idx, flow_vel_quad_idx);

        FECellIntegrator<dim, dim, number> recoil_pressure_eval(matrix_free,
                                                                flow_vel_dof_idx,
                                                                flow_vel_quad_idx);

        const unsigned int cell_category = cut_type == CutUtil::CutPhaseType::not_cut ?
                                             0 :
                                             matrix_free.get_cell_range_category(cell_range);

        std::vector<FECellIntegrator<dim, 1, number>> temperature_eval;
        if (cut_type == CutUtil::CutPhaseType::not_cut)
          temperature_eval.emplace_back(matrix_free, heat_dof_idx, flow_vel_quad_idx);
        else // temperature is cut
          {
            if (cell_category == CutUtil::CellCategory::liquid or
                cell_category == CutUtil::CellCategory::intersected)
              temperature_eval.emplace_back(matrix_free,
                                            heat_dof_idx,
                                            flow_vel_quad_idx,
                                            0 /*selected component*/,
                                            cell_category /*active_fe_index*/);
            if (cut_type == CutUtil::CutPhaseType::two_phase_cut and
                (cell_category == CutUtil::CellCategory::gas or
                 cell_category == CutUtil::CellCategory::intersected))
              temperature_eval.emplace_back(matrix_free,
                                            heat_dof_idx,
                                            flow_vel_quad_idx,
                                            1 /*selected component*/,
                                            cell_category /*active_fe_index*/);
          }

        FECellIntegrator<dim, 1, number> heaviside_interpolated_to_pressure_space_eval(
          matrix_free, flow_pressure_dof_idx, flow_vel_quad_idx);

        std::unique_ptr<FECellIntegrator<dim, 1, number>> evapor_flux_val;
        if (model_type == RecoilPressureModelType::hybrid)
          evapor_flux_val = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                               evapor_dof_idx,
                                                                               flow_vel_quad_idx);

        auto &used_heaviside_eval = do_level_set_pressure_gradient_interpolation ?
                                      heaviside_interpolated_to_pressure_space_eval :
                                      heaviside_eval;

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            heaviside_eval.reinit(cell);
            heaviside_eval.read_dof_values_plain(level_set_as_heaviside);

            if (do_level_set_pressure_gradient_interpolation)
              {
                heaviside_interpolated_to_pressure_space_eval.reinit(cell);

                DoFTools::compute_gradient_at_interpolated_dof_values<dim>(
                  heaviside_eval,
                  heaviside_interpolated_to_pressure_space_eval,
                  ls_to_pressure_grad_interpolation_matrix);
              }

            if (delta_phase_weighted or (cut_type != CutUtil::CutPhaseType::not_cut and
                                         cell_category == CutUtil::CellCategory::intersected))
              used_heaviside_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            else
              used_heaviside_eval.evaluate(EvaluationFlags::gradients);

            for (auto &heat_eval : temperature_eval)
              {
                heat_eval.reinit(cell);
                heat_eval.read_dof_values_plain(temperature);
                heat_eval.evaluate(EvaluationFlags::values);
              }

            if (evapor_flux_val)
              {
                evapor_flux_val->reinit(cell);
                evapor_flux_val->read_dof_values_plain(evaporative_mass_flux);
                evapor_flux_val->evaluate(EvaluationFlags::values);
              }

            recoil_pressure_eval.reinit(cell);

            for (const unsigned int q : recoil_pressure_eval.quadrature_point_indices())
              {
                VectorizedArray<number> temp;
                switch (cut_type)
                  {
                      case CutUtil::CutPhaseType::not_cut: {
                        temp = temperature_eval[0].get_value(q);
                        break;
                      }
                      case CutUtil::CutPhaseType::two_phase_cut: {
                        if (cell_category == CutUtil::CellCategory::intersected)
                          // For intersected cut cells, temperature_val[0] contains the temperature
                          // values of the liquid domain and temperature_val[1] the temperature
                          // values of the gas domain that each may be ghost values. We use the
                          // level set as heaviside as in indicator to select the non-ghosted values
                          // each.
                          temp = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                            used_heaviside_eval.get_value(q),
                            0.5,
                            temperature_eval[0].get_value(q),
                            temperature_eval[1].get_value(q));
                        else
                          temp = temperature_eval[0].get_value(q);
                        break;
                      }
                      case CutUtil::CutPhaseType::one_phase_cut: {
                        if (cell_category == CutUtil::CellCategory::liquid)
                          temp = temperature_eval[0].get_value(q);
                        else if (cell_category == CutUtil::CellCategory::intersected)
                          // For intersected cut cells, temperature_val[0] contains the temperature
                          // values of the liquid domain that each may be ghost values. We use the
                          // level set as heaviside as in indicator to select the non-ghosted values
                          // each. For ghosted values we insert a dummy temperature.
                          temp = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                            used_heaviside_eval.get_value(q),
                            0.5,
                            temperature_eval[0].get_value(q),
                            dummy_temperature);
                        else if (cell_category == CutUtil::CellCategory::gas)
                          // we cannot evaluate temperature here so we just use a dummy value
                          temp = dummy_temperature;
                        else
                          DEAL_II_NOT_IMPLEMENTED();
                        break;
                      }
                    default:
                      DEAL_II_NOT_IMPLEMENTED();
                  }

                VectorizedArray<number> m_dot = 0.0;
                if (evapor_flux_val)
                  m_dot = evapor_flux_val->get_value(q);

                VectorizedArray<number> weight = 1.0;
                if (delta_phase_weighted)
                  weight = delta_phase_weighted->compute_weight(used_heaviside_eval.get_value(q));

                const VectorizedArray<number> recoil_pressure_coefficient =
                  evapor_flux_val ?
                    recoil_pressure_model->compute_recoil_pressure_coefficient(temp,
                                                                               m_dot,
                                                                               1. / weight) :
                    recoil_pressure_model->compute_recoil_pressure_coefficient(temp);


                recoil_pressure_eval.submit_value(recoil_pressure_coefficient *
                                                    used_heaviside_eval.get_gradient(q) * weight,
                                                  q);
              }
            recoil_pressure_eval.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      level_set_as_heaviside,
      zero_out);
  }

  template class RecoilPressureOperation<1, double>;
  template class RecoilPressureOperation<2, double>;
  template class RecoilPressureOperation<3, double>;

  template class RecoilPressurePhenomenologicalModel<double>;
  template class RecoilPressureHybridModel<double>;
} // namespace MeltPoolDG::Evaporation
