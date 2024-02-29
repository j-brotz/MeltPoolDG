/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, June 2021
 *
 * ---------------------------------------------------------------------*/
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/evaporation/evaporation_source_terms_continuous.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**
   * TODO: DOCU
   */
  template <int dim>
  EvaporationSourceTermsContinuous<dim>::EvaporationSourceTermsContinuous(
    const ScratchData<dim>                      &scratch_data,
    const EvaporationData<double>               &evapor_data,
    const VectorType                            &level_set_as_heaviside,
    const BlockVectorType                       &normal_vector,
    const VectorType                            &evaporative_mass_flux,
    const unsigned int                           ls_hanging_nodes_dof_idx,
    const unsigned int                           ls_quad_idx,
    const unsigned int                           normal_dof_idx,
    const unsigned int                           evapor_vel_dof_idx,
    const unsigned int                           evapor_mass_flux_dof_idx,
    const double                                 tolerance_normal_vector,
    const double                                 density_vapor,
    const double                                 density_liquid,
    const TwoPhaseFluidPropertiesTransitionType &two_phase_properties_transition_type)
    : scratch_data(scratch_data)
    , evapor_data(evapor_data)
    , level_set_as_heaviside(level_set_as_heaviside)
    , normal_vector(normal_vector)
    , evaporative_mass_flux(evaporative_mass_flux)
    , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx)
    , ls_quad_idx(ls_quad_idx)
    , normal_dof_idx(normal_dof_idx)
    , evapor_vel_dof_idx(evapor_vel_dof_idx)
    , evapor_mass_flux_dof_idx(evapor_mass_flux_dof_idx)
    , tolerance_normal_vector(tolerance_normal_vector)
    , density_vapor(density_vapor)
    , density_liquid(density_liquid)
    , two_phase_properties_transition_type(two_phase_properties_transition_type)
  {}

  template <int dim>
  void
  EvaporationSourceTermsContinuous<dim>::compute_level_set_source_term(
    VectorType        &level_set_source_term,
    const unsigned int ls_dof_idx,
    const VectorType  &level_set,
    const unsigned int pressure_dof_idx)
  {
    if (evapor_data.do_level_set_pressure_gradient_interpolation &&
        ls_to_pressure_grad_interpolation_matrix.m() == 0) // do only once
      ls_to_pressure_grad_interpolation_matrix =
        UtilityFunctions::create_dof_interpolation_matrix<dim>(
          scratch_data.get_dof_handler(pressure_dof_idx),
          scratch_data.get_dof_handler(ls_dof_idx),
          true /*do sort lexicographic (matrix-free)*/);

    const bool update_ghosts = !level_set.has_ghost_elements();
    if (update_ghosts)
      level_set.update_ghost_values();

    const bool evapor_update_ghosts = !evaporative_mass_flux.has_ghost_elements();
    if (evapor_update_ghosts)
      evaporative_mass_flux.update_ghost_values();

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto       &level_set_source_term,
          const auto &level_set_as_heaviside,
          auto        macro_cells) {
        FECellIntegrator<dim, 1, double> ls(matrix_free, ls_dof_idx, ls_quad_idx);

        FECellIntegrator<dim, 1, double> hs(matrix_free, ls_hanging_nodes_dof_idx, ls_quad_idx);

        FECellIntegrator<dim, 1, double> evap_flux(matrix_free,
                                                   evapor_mass_flux_dof_idx,
                                                   ls_quad_idx);

        FECellIntegrator<dim, 1, double> interpolated_level_set_to_pressure_space(matrix_free,
                                                                                  pressure_dof_idx,
                                                                                  ls_quad_idx);

        auto &used_level_set = evapor_data.do_level_set_pressure_gradient_interpolation ?
                                 interpolated_level_set_to_pressure_space :
                                 ls;

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            ls.reinit(cell);
            ls.read_dof_values_plain(level_set);

            if (evapor_data.do_level_set_pressure_gradient_interpolation)
              {
                interpolated_level_set_to_pressure_space.reinit(cell);

                UtilityFunctions::compute_gradient_at_interpolated_dof_values<dim>(
                  ls,
                  interpolated_level_set_to_pressure_space,
                  ls_to_pressure_grad_interpolation_matrix);
              }

            used_level_set.evaluate(EvaluationFlags::gradients);

            hs.reinit(cell);
            hs.read_dof_values_plain(level_set_as_heaviside);
            hs.evaluate(EvaluationFlags::values);

            evap_flux.reinit(cell);
            evap_flux.read_dof_values_plain(evaporative_mass_flux);
            evap_flux.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
              {
                ls.submit_value(-evap_flux.get_value(q_index) *
                                  used_level_set.get_gradient(q_index).norm() *
                                  LevelSet::Tools::interpolate(hs.get_value(q_index),
                                                               1. / density_vapor,
                                                               1. / density_liquid),
                                q_index);
              }
            ls.integrate_scatter(EvaluationFlags::values, level_set_source_term);
          }
      },
      level_set_source_term,
      level_set_as_heaviside,
      true /*zero_out*/);

    if (evapor_update_ghosts)
      evaporative_mass_flux.zero_out_ghost_values();
    if (update_ghosts)
      level_set.zero_out_ghost_values();
  }

  template <int dim>
  void
  EvaporationSourceTermsContinuous<dim>::compute_evaporation_velocity(
    VectorType &evaporation_velocity)
  {
    AssertThrow(!evapor_data.do_level_set_pressure_gradient_interpolation, ExcNotImplemented());

    const bool update_ghosts = !level_set_as_heaviside.has_ghost_elements();
    if (update_ghosts)
      level_set_as_heaviside.update_ghost_values();

    const bool normal_update_ghosts = !normal_vector.has_ghost_elements();
    if (normal_update_ghosts)
      normal_vector.update_ghost_values();

    const bool evapor_update_ghosts = !evaporative_mass_flux.has_ghost_elements();
    if (evapor_update_ghosts)
      evaporative_mass_flux.update_ghost_values();

    FECellIntegrator<dim, 1, double> ls(scratch_data.get_matrix_free(),
                                        ls_hanging_nodes_dof_idx,
                                        ls_quad_idx);

    FECellIntegrator<dim, dim, double> normal_vec(scratch_data.get_matrix_free(),
                                                  normal_dof_idx,
                                                  ls_quad_idx);

    FECellIntegrator<dim, 1, double> evap_flux(scratch_data.get_matrix_free(),
                                               evapor_mass_flux_dof_idx,
                                               ls_quad_idx);

    evaporation_velocities.resize(scratch_data.get_matrix_free().n_cell_batches() * ls.n_q_points);

    for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
      {
        Tensor<1, dim, VectorizedArray<double>> *evapor_vel = begin_evaporation_velocity(cell);

        ls.reinit(cell);
        ls.read_dof_values_plain(level_set_as_heaviside);
        ls.evaluate(EvaluationFlags::values);

        normal_vec.reinit(cell);
        normal_vec.read_dof_values_plain(normal_vector);
        normal_vec.evaluate(EvaluationFlags::values);

        evap_flux.reinit(cell);
        evap_flux.read_dof_values_plain(evaporative_mass_flux);
        evap_flux.evaluate(EvaluationFlags::values);

        for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
          {
            const auto n_phi =
              MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index),
                                                      tolerance_normal_vector);

            //              ρ
            // evaluate  ------
            //            dρ/dΦ
            VectorizedArray<double> rho_d_rho_d_phi = 1.0;

            if (two_phase_properties_transition_type ==
                TwoPhaseFluidPropertiesTransitionType::smooth)
              {
                // clang-format off
                  rho_d_rho_d_phi = (ls.get_value(q_index) * density_liquid + (1.-ls.get_value(q_index) * density_vapor)) 
                    / //-----------------------------------------------------------------------------------------------------
                                                    (density_liquid - density_vapor);
                // clang-format on
              }

            evapor_vel[q_index] = n_phi * evap_flux.get_value(q_index) * rho_d_rho_d_phi *
                                  LevelSet::Tools::interpolate(ls.get_value(q_index),
                                                               1. / density_vapor,
                                                               1. / density_liquid);
          }
      }
    if (update_ghosts)
      level_set_as_heaviside.zero_out_ghost_values();
    if (normal_update_ghosts)
      normal_vector.zero_out_ghost_values();
    if (evapor_update_ghosts)
      evaporative_mass_flux.zero_out_ghost_values();

    scratch_data.initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);

    /**
     * write interface velocity to dof vector
     */
    if (scratch_data.is_hex_mesh())
      MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, dim>(
        evaporation_velocity,
        scratch_data.get_matrix_free(),
        evapor_vel_dof_idx,
        ls_quad_idx,
        [&](const unsigned int cell,
            const unsigned int quad) -> const Tensor<1, dim, VectorizedArray<double>> & {
          return begin_evaporation_velocity(cell)[quad];
        });

    scratch_data.get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);

    Journal::print_formatted_norm(
      scratch_data.get_pcout(1),
      [&]() -> double {
        return VectorTools::compute_norm<dim>(evaporation_velocity,
                                              scratch_data,
                                              evapor_vel_dof_idx,
                                              ls_quad_idx);
      },
      "evaporative_velocity",
      "evaporation_operation",
      10);
    evaporation_velocity.zero_out_ghost_values();
  }

  template <int dim>
  void
  EvaporationSourceTermsContinuous<dim>::compute_mass_balance_source_term(
    VectorType        &mass_balance_source_term,
    const unsigned int pressure_dof_idx,
    const unsigned int pressure_quad_idx,
    bool               zero_out)
  {
    if (evapor_data.do_level_set_pressure_gradient_interpolation &&
        ls_to_pressure_grad_interpolation_matrix.m() == 0) // do only once
      ls_to_pressure_grad_interpolation_matrix =
        UtilityFunctions::create_dof_interpolation_matrix<dim>(
          scratch_data.get_dof_handler(pressure_dof_idx),
          scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
          true /*do sort lexicographic (matrix-free)*/);

    const bool evapor_update_ghosts = !evaporative_mass_flux.has_ghost_elements();
    if (evapor_update_ghosts)
      evaporative_mass_flux.update_ghost_values();

    double mass = 0.0;

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto       &mass_balance_source_term,
          const auto &level_set_as_heaviside,
          auto        macro_cells) {
        FECellIntegrator<dim, 1, double> heaviside(matrix_free,
                                                   ls_hanging_nodes_dof_idx,
                                                   pressure_quad_idx);

        FECellIntegrator<dim, 1, double> mass_flux(matrix_free,
                                                   pressure_dof_idx,
                                                   pressure_quad_idx);

        FECellIntegrator<dim, 1, double> evap_flux(scratch_data.get_matrix_free(),
                                                   evapor_mass_flux_dof_idx,
                                                   pressure_quad_idx);

        FECellIntegrator<dim, 1, double> interpolated_level_set_to_pressure_space(
          matrix_free, pressure_dof_idx, pressure_quad_idx);

        auto &used_level_set = evapor_data.do_level_set_pressure_gradient_interpolation ?
                                 interpolated_level_set_to_pressure_space :
                                 heaviside;

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            heaviside.reinit(cell);
            heaviside.read_dof_values_plain(level_set_as_heaviside);

            if (evapor_data.do_level_set_pressure_gradient_interpolation)
              {
                interpolated_level_set_to_pressure_space.reinit(cell);

                UtilityFunctions::compute_gradient_at_interpolated_dof_values<dim>(
                  heaviside,
                  interpolated_level_set_to_pressure_space,
                  ls_to_pressure_grad_interpolation_matrix);
              }

            used_level_set.evaluate(EvaluationFlags::gradients);

            mass_flux.reinit(cell);

            evap_flux.reinit(cell);
            evap_flux.read_dof_values_plain(evaporative_mass_flux);
            evap_flux.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < mass_flux.n_q_points; ++q_index)
              {
                mass_flux.submit_value((1. / density_liquid - 1. / density_vapor) *
                                         used_level_set.get_gradient(q_index).norm() *
                                         evap_flux.get_value(q_index),
                                       q_index);
                // compute overall rhs
                for (unsigned int v = 0;
                     v < scratch_data.get_matrix_free().n_active_entries_per_cell_batch(cell);
                     ++v)
                  {
                    mass += (1. / density_liquid - 1. / density_vapor) *
                            used_level_set.get_gradient(q_index).norm()[v] *
                            evap_flux.get_value(q_index)[v] * mass_flux.JxW(q_index)[v];
                  }
              }

            mass_flux.integrate_scatter(EvaluationFlags::values, mass_balance_source_term);
          }
      },
      mass_balance_source_term,
      level_set_as_heaviside,
      zero_out);
    if (evapor_update_ghosts)
      evaporative_mass_flux.zero_out_ghost_values();

    std::ostringstream str;
    str << "evaporation: jump in the velocity field = "
        << Utilities::MPI::sum(mass, scratch_data.get_mpi_comm());

    Journal::print_line(scratch_data.get_pcout(0),
                        str.str(),
                        "evaporation_source_terms_continuous");

    Journal::print_formatted_norm(scratch_data.get_pcout(0),
                                  mass_balance_source_term.l2_norm(),
                                  "evapartive_mass_source",
                                  "evaporation_source_terms_continuous",
                                  10);
  }

  template <int dim>
  void
  EvaporationSourceTermsContinuous<dim>::compute_heat_source_term(
    [[maybe_unused]] VectorType &heat_source_term)
  {
    AssertThrow(false, ExcNotImplemented());
  }

  template <int dim>
  inline Tensor<1, dim, VectorizedArray<double>> *
  EvaporationSourceTermsContinuous<dim>::begin_evaporation_velocity(const unsigned int macro_cell)
  {
    AssertIndexRange(macro_cell, scratch_data.get_matrix_free().n_cell_batches());
    return &evaporation_velocities[scratch_data.get_matrix_free().get_n_q_points(ls_quad_idx) *
                                   macro_cell];
  }

  template <int dim>
  inline const Tensor<1, dim, VectorizedArray<double>> &
  EvaporationSourceTermsContinuous<dim>::begin_evaporation_velocity(
    const unsigned int macro_cell) const
  {
    AssertIndexRange(macro_cell, scratch_data.get_matrix_free().n_cell_batches());
    return evaporation_velocities[scratch_data.get_matrix_free().get_n_q_points(ls_quad_idx) *
                                  macro_cell];
  }
  template class EvaporationSourceTermsContinuous<1>;
  template class EvaporationSourceTermsContinuous<2>;
  template class EvaporationSourceTermsContinuous<3>;
} // namespace MeltPoolDG::Evaporation
