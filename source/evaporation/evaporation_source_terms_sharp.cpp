#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/evaporation/evaporation_source_terms_sharp.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  template <int dim>
  EvaporationSourceTermsSharp<dim>::EvaporationSourceTermsSharp(
    const ScratchData<dim> &       scratch_data,
    const EvaporationData<double> &evapor_data,
    const VectorType &             level_set_as_heaviside,
    const BlockVectorType &        normal_vector,
    const VectorType &             evaporative_mass_flux,
    const unsigned int             ls_hanging_nodes_dof_idx,
    const unsigned int             ls_quad_idx,
    const unsigned int             normal_dof_idx,
    const unsigned int             evapor_vel_dof_idx,
    const double                   tolerance_normal_vector,
    const double                   density_vapor,
    const double                   density_liquid)
    : scratch_data(scratch_data)
    , evapor_data(evapor_data)
    , level_set_as_heaviside(level_set_as_heaviside)
    , normal_vector(normal_vector)
    , evaporative_mass_flux(evaporative_mass_flux)
    , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx)
    , ls_quad_idx(ls_quad_idx)
    , normal_dof_idx(normal_dof_idx)
    , evapor_vel_dof_idx(evapor_vel_dof_idx)
    , tolerance_normal_vector(tolerance_normal_vector)
    , density_vapor(density_vapor)
    , density_liquid(density_liquid)
  {}

  template <int dim>
  void
  EvaporationSourceTermsSharp<dim>::compute_evaporation_velocity(
    VectorType &                        evaporation_velocity,
    [[maybe_unused]] const std::string &interpolation_type_parameters)
  {
    /**
     * evaporation velocity at quadrature points
     */
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaporation_velocities;

    level_set_as_heaviside.update_ghost_values();
    normal_vector.update_ghost_values();
    evaporative_mass_flux.update_ghost_values();

    FECellIntegrator<dim, 1, double> ls(scratch_data.get_matrix_free(),
                                        ls_hanging_nodes_dof_idx,
                                        ls_quad_idx);

    FECellIntegrator<dim, dim, double> normal_vec(scratch_data.get_matrix_free(),
                                                  normal_dof_idx,
                                                  ls_quad_idx);

    FECellIntegrator<dim, 1, double> evap_flux(
      scratch_data.get_matrix_free(),
      ls_hanging_nodes_dof_idx, // @todo: generalize --> temp_dof_idx
      ls_quad_idx);

    evaporation_velocities.resize(scratch_data.get_matrix_free().n_cell_batches() * ls.n_q_points);

    for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
      {
        Tensor<1, dim, VectorizedArray<double>> *evapor_vel =
          &evaporation_velocities[scratch_data.get_matrix_free().get_n_q_points(ls_quad_idx) *
                                  cell];

        ls.reinit(cell);
        ls.read_dof_values(level_set_as_heaviside);
        ls.evaluate(true, true);

        normal_vec.reinit(cell);
        normal_vec.read_dof_values(normal_vector);
        normal_vec.evaluate(EvaluationFlags::values);

        evap_flux.reinit(cell);
        evap_flux.read_dof_values(evaporative_mass_flux);
        evap_flux.evaluate(EvaluationFlags::values);

        for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
          {
            const auto n_phi =
              MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index));

            auto is_liquid = compare_and_apply_mask<SIMDComparison::less_than>(
              ls.get_value(q_index),
              VectorizedArray<double>(0.5),
              evap_flux.get_value(q_index) / density_vapor,
              evap_flux.get_value(q_index) / density_liquid);

            evapor_vel[q_index] = is_liquid * n_phi;
          }
      }
    level_set_as_heaviside.zero_out_ghost_values();
    normal_vector.zero_out_ghost_values();
    evaporative_mass_flux.zero_out_ghost_values();

    scratch_data.initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);

    /**
     * write interface velocity to dof vector
     */
    if (scratch_data.is_hex_mesh())
      UtilityFunctions::fill_dof_vector_from_cell_operation_vec<dim, dim>(
        evaporation_velocity,
        scratch_data.get_matrix_free(),
        evapor_vel_dof_idx,
        ls_quad_idx,
        [&](const unsigned int cell,
            const unsigned int quad) -> const Tensor<1, dim, VectorizedArray<double>> & {
          return const_cast<const Tensor<1, dim, VectorizedArray<double>> &>(
            evaporation_velocities[scratch_data.get_matrix_free().get_n_q_points(ls_quad_idx) *
                                     cell +
                                   quad]);
        });

    scratch_data.get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);
    scratch_data.get_pcout() << "    | evapor: |u|2 = " << evaporation_velocity.l2_norm()
                             << std::endl;

    evaporation_velocity.zero_out_ghost_values();
  }

  template <int dim>
  void
  EvaporationSourceTermsSharp<dim>::compute_mass_balance_source_term(
    VectorType &                        mass_balance_source_term,
    [[maybe_unused]] const unsigned int pressure_dof_idx,
    [[maybe_unused]] const unsigned int pressure_quad_idx,
    [[maybe_unused]] bool               zero_out)
  {
    if constexpr (dim > 1) // @todo: otherwise i am getting a compiling error atm
      {
        evaporative_mass_flux.update_ghost_values();
        level_set_as_heaviside.update_ghost_values();

        FEPointEvaluation<1, dim> mass_flux(
          scratch_data.get_mapping(),
          scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).get_fe(),
          update_values);
        FEPointEvaluation<1, dim> rhs_continuity(
          scratch_data.get_mapping(),
          scratch_data.get_dof_handler(pressure_dof_idx).get_fe(),
          update_values);

        std::vector<double>                  buffer;
        std::vector<types::global_dof_index> local_dof_indices;

        UtilityFunctions::evaluate_at_interface<dim>(
          scratch_data.get_dof_handler(pressure_dof_idx),
          scratch_data.get_mapping(),
          level_set_as_heaviside,
          [&](const auto &cell, const auto &, const auto &points, const auto &weights) {
            // evaluate rhs term
            local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
            buffer.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            const unsigned int n_points = points.size();

            const ArrayView<const Point<dim>> unit_points(points.data(), n_points);
            const ArrayView<const double>     JxW(weights.data(), n_points);

            mass_flux.reinit(cell, unit_points);
            rhs_continuity.reinit(cell, unit_points);

            // gather mass_flux
            scratch_data.get_constraint(ls_hanging_nodes_dof_idx)
              .get_dof_values(evaporative_mass_flux,
                              local_dof_indices.begin(),
                              buffer.begin(),
                              buffer.end());

            // evaluate mass_flux
            mass_flux.evaluate(buffer, EvaluationFlags::values);

            for (unsigned int q = 0; q < n_points; ++q)
              {
                rhs_continuity.submit_value(mass_flux.get_value(q) *
                                              (1. / density_liquid - 1. / density_vapor) * JxW[q],
                                            q);
              }

            // integrate rhs term of the continuity equation
            rhs_continuity.integrate(buffer, EvaluationFlags::values);

            scratch_data.get_constraint(ls_hanging_nodes_dof_idx)
              .distribute_local_to_global(buffer, local_dof_indices, mass_balance_source_term);
          },
          0.5, /*contour value*/
          3 /*n_subdivisions*/);
        mass_balance_source_term.compress(VectorOperation::add);
        mass_balance_source_term.update_ghost_values();
        evaporative_mass_flux.zero_out_ghost_values();
        level_set_as_heaviside.zero_out_ghost_values();
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }

  template <int dim>
  void
  EvaporationSourceTermsSharp<dim>::compute_heat_source_term(
    [[maybe_unused]] VectorType &heat_source_term)
  {
    AssertThrow(false, ExcNotImplemented());
  }

  template class EvaporationSourceTermsSharp<1>;
  template class EvaporationSourceTermsSharp<2>;
  template class EvaporationSourceTermsSharp<3>;
} // namespace MeltPoolDG::Evaporation
