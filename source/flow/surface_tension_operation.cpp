#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/flow/surface_tension_operation.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <cmath>

namespace MeltPoolDG::Flow
{
  template <int dim>
  SurfaceTensionOperation<dim>::SurfaceTensionOperation(
    const SurfaceTensionData<double> &data_in,
    const ScratchData<dim>           &scratch_data,
    const VectorType                 &level_set_as_heaviside,
    const VectorType                 &solution_curvature,
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
    , alpha_residual(data.surface_tension_coefficient * data.coefficient_residual_fraction)
  {
    AssertThrow(data.time_step_limit.scale_factor >= 0 && data.time_step_limit.scale_factor <= 1.0,
                ExcMessage(
                  "The scale factor for the time step limit must be between 0 and 1.0. Abort..."));

    if (do_level_set_pressure_gradient_interpolation)
      {
        ls_to_pressure_grad_interpolation_matrix =
          UtilityFunctions::create_dof_interpolation_matrix<dim>(
            scratch_data.get_dof_handler(flow_pressure_hanging_nodes_dof_idx),
            scratch_data.get_dof_handler(ls_dof_idx),
            true);
      }

    delta_phase_weighted =
      create_phase_weighted_delta_approximation(data.delta_approximation_phase_weighted);

    //@todo add assert for parameters
  }



  template <int dim>
  void
  SurfaceTensionOperation<dim>::register_temperature_and_normal_vector(
    const unsigned int     temp_dof_idx_in,
    const unsigned int     normal_dof_idx_in,
    const VectorType      *temperature_in,
    const BlockVectorType *solution_normal_vector_in)
  {
    temp_dof_idx           = temp_dof_idx_in;
    normal_dof_idx         = normal_dof_idx_in;
    temperature            = temperature_in;
    solution_normal_vector = solution_normal_vector_in;
    AssertThrow(
      data.reference_temperature > numbers::invalid_double,
      ExcMessage(
        "For temperature-dependent surface tension, a reference temperature needs to be defined. Abort..."));
  }



  template <int dim>
  void
  SurfaceTensionOperation<dim>::register_solid_fraction(const unsigned int solid_dof_idx_in,
                                                        const VectorType  *solid_in)
  {
    solid_dof_idx = solid_dof_idx_in;
    solid         = solid_in;
  }



  template <int dim>
  void
  SurfaceTensionOperation<dim>::compute_surface_tension(VectorType &force_rhs, const bool zero_out)
  {
    const bool curv_update_ghosts = !solution_curvature.has_ghost_elements();

    if (curv_update_ghosts)
      solution_curvature.update_ghost_values();

    bool normal_update_ghosts      = true;
    bool temperature_update_ghosts = true;
    bool solid_update_ghosts       = true;

    if (temperature)
      {
        normal_update_ghosts = !solution_normal_vector->has_ghost_elements();

        if (normal_update_ghosts)
          solution_normal_vector->update_ghost_values();

        temperature_update_ghosts = !temperature->has_ghost_elements();
        if (temperature_update_ghosts)
          temperature->update_ghost_values();
      }
    if (solid)
      {
        solid_update_ghosts = !solid->has_ghost_elements();
        if (solid_update_ghosts)
          solid->update_ghost_values();
      }

    const double tolerance_normal_vector =
      UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                            scratch_data.get_mapping());

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto       &force_rhs,
          const auto &level_set_as_heaviside,
          auto        macro_cells) {
        FECellIntegrator<dim, 1, double> level_set(matrix_free, ls_dof_idx, flow_vel_quad_idx);

        FECellIntegrator<dim, 1, double> curvature(matrix_free, curv_dof_idx, flow_vel_quad_idx);

        FECellIntegrator<dim, 1, double> interpolated_level_set_to_pressure_space(
          matrix_free, flow_pressure_hanging_nodes_dof_idx, flow_vel_quad_idx);

        std::unique_ptr<FECellIntegrator<dim, dim, double>> normal_vec;
        std::unique_ptr<FECellIntegrator<dim, 1, double>>   temperature_val;
        std::unique_ptr<FECellIntegrator<dim, 1, double>>   solid_val;

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
        if (solid)
          solid_val = std::make_unique<FECellIntegrator<dim, 1, double>>(matrix_free,
                                                                         solid_dof_idx,
                                                                         flow_vel_quad_idx);

        FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                           flow_vel_dof_idx,
                                                           flow_vel_quad_idx);

        auto alpha = VectorizedArray<double>(data.surface_tension_coefficient);

        const double &d_alpha0 = data.temperature_dependent_surface_tension_coefficient;

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
            if (solid)
              {
                solid_val->reinit(cell);
                solid_val->read_dof_values_plain(*solid);
                solid_val->evaluate(EvaluationFlags::values);
              }

            for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
              {
                const auto mask =
                  solid ? 1. - solid_val->get_value(q_index) : VectorizedArray<double>(1.0);

                VectorizedArray<double> weight(1.0);
                if (delta_phase_weighted)
                  weight = delta_phase_weighted->compute_weight(used_level_set.get_value(q_index));

                if (temperature)
                  {
                    const auto n =
                      MeltPoolDG::VectorTools::normalize<dim>(normal_vec->get_value(q_index),
                                                              tolerance_normal_vector);
                    const auto T      = temperature_val->get_value(q_index);
                    const auto grad_T = temperature_val->get_gradient(q_index);

                    // compute constant surface tension
                    alpha = local_compute_temperature_dependent_surface_tension_coefficient(T);
                    const auto constant_surface_tension = alpha * curvature.get_value(q_index) *
                                                          weight *
                                                          used_level_set.get_gradient(q_index);


                    // compute Marangoni convection
                    const auto delta = used_level_set.get_gradient(q_index).norm() * weight;
                    const Tensor<1, dim, VectorizedArray<double>> temp_surf_ten =
                      -d_alpha0 * (grad_T - n * scalar_product(n, grad_T)) * delta;

                    surface_tension.submit_value(mask * (constant_surface_tension + temp_surf_ten),
                                                 q_index);
                  }
                else
                  surface_tension.submit_value(mask * alpha * curvature.get_value(q_index) *
                                                 used_level_set.get_gradient(q_index) * weight,
                                               q_index);
              }
            surface_tension.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      level_set_as_heaviside,
      zero_out);

    if (curv_update_ghosts)
      solution_curvature.zero_out_ghost_values();

    if (temperature)
      {
        if (temperature_update_ghosts)
          temperature->zero_out_ghost_values();

        if (normal_update_ghosts)
          solution_normal_vector->zero_out_ghost_values();
      }
    if (solid)
      {
        if (solid_update_ghosts)
          solid->zero_out_ghost_values();
      }
  }

  template <int dim>
  double
  SurfaceTensionOperation<dim>::compute_time_step_limit(const double density_1,
                                                        const double density_2)
  {
    double alpha = data.surface_tension_coefficient;

    // compute maximum value for alpha
    if (temperature)
      {
        // Surface tension coefficient decreases with increasing temperature --> the maximum
        // surface tension coefficient arises at the minimum temperature.
        if (data.temperature_dependent_surface_tension_coefficient > 0)
          {
            const double T_min =
              VectorTools::min_element(*temperature, scratch_data.get_mpi_comm());
            alpha = local_compute_temperature_dependent_surface_tension_coefficient(T_min);
          }
        // Surface tension coefficient increases with decreasing temperature --> the maximum
        // surface tension coefficient arises at the maximum temperature.
        else
          {
            const double T_max =
              VectorTools::max_element(*temperature, scratch_data.get_mpi_comm());
            alpha = local_compute_temperature_dependent_surface_tension_coefficient(T_max);
          }
      }

    return data.time_step_limit.scale_factor *
           std::sqrt((density_1 + density_2) *
                     Utilities::fixed_power<3>(scratch_data.get_min_cell_size()) /
                     (2 * numbers::PI * alpha));
  }

  template <int dim>
  template <typename number>
  number
  SurfaceTensionOperation<dim>::local_compute_temperature_dependent_surface_tension_coefficient(
    const number &T)
  {
    const number &alpha0   = data.surface_tension_coefficient;
    const number &d_alpha0 = data.temperature_dependent_surface_tension_coefficient;
    const number &T0       = data.reference_temperature;

    auto alpha = alpha0 - d_alpha0 * (T - T0);

    // The surface tension must not become negative or smaller than its residual
    // value.
    alpha = compare_and_apply_mask<SIMDComparison::less_than>(alpha,
                                                              alpha_residual,
                                                              alpha_residual,
                                                              alpha);
    return alpha;
  }


  template class SurfaceTensionOperation<1>;
  template class SurfaceTensionOperation<2>;
  template class SurfaceTensionOperation<3>;
} // namespace MeltPoolDG::Flow
