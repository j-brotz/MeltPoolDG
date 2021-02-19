/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim>
  class SurfaceTensionOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    //@todo: merge the two following functions

    /*
     *  temperature-independent surface tension
     */
    static void
    compute_surface_tension(VectorType &            force_rhs,
                            const ScratchData<dim> &scratch_data,
                            const VectorType &      level_set_as_heaviside,
                            const VectorType &      curvature_vec,
                            const double            surface_tension_coefficient,
                            const unsigned int      ls_dof_idx,
                            const unsigned int      curv_dof_idx,
                            const unsigned int      flow_vel_dof_idx,
                            const unsigned int      flow_quad_idx,
                            const bool              zero_out = true)
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
                                                             flow_vel_dof_idx,
                                                             flow_quad_idx);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              level_set.reinit(cell);
              level_set.read_dof_values_plain(level_set_as_heaviside);
              level_set.evaluate(false, true);

              surface_tension.reinit(cell);

              curvature.reinit(cell);
              curvature.gather_evaluate(curvature_vec, true, false);

              for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
                {
                  surface_tension.submit_value(surface_tension_coefficient *
                                                 level_set.get_gradient(q_index) *
                                                 curvature.get_value(q_index),
                                               q_index);
                }
              surface_tension.integrate_scatter(true, false, force_rhs);
            }
        },
        force_rhs,
        level_set_as_heaviside,
        zero_out);

      curvature_vec.zero_out_ghosts();
    }
    /**
     *  This function introduces the basic framework for temperature-dependent surface tension
     *  forces, i.e. Marangoni convection.
     */
    static void
    compute_temperature_dependent_surface_tension(
      ScratchData<dim>   scratch_data,
      VectorType &       force_rhs,
      const VectorType & level_set_as_heaviside,
      const VectorType & solution_curvature,
      const VectorType & temperature,
      const double       surface_tension_coefficient,
      const double       temperature_dependent_surface_tension_coefficient,
      const double       surface_tension_reference_temperature,
      const unsigned int ls_dof_idx,
      const unsigned int flow_vel_dof_idx,
      const unsigned int flow_vel_quad_idx,
      const unsigned int temp_dof_idx,
      const bool         zero_out = true)
    {
      solution_curvature.update_ghost_values();

      scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
        [&](const auto &matrix_free,
            auto &      force_rhs,
            const auto &level_set_as_heaviside,
            auto        macro_cells) {
          FECellIntegrator<dim, 1, double> level_set(matrix_free, ls_dof_idx, flow_vel_quad_idx);

          FECellIntegrator<dim, 1, double> curvature(
            matrix_free, temp_dof_idx, flow_vel_quad_idx); /*@todo: own index for curvature*/

          FECellIntegrator<dim, 1, double> temperature_val(matrix_free,
                                                           temp_dof_idx,
                                                           flow_vel_quad_idx);

          FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                             flow_vel_dof_idx,
                                                             flow_vel_quad_idx);

          const double &alpha0   = surface_tension_coefficient;
          const double &d_alpha0 = temperature_dependent_surface_tension_coefficient;
          const auto    T0       = VectorizedArray<double>(surface_tension_reference_temperature);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              level_set.reinit(cell);
              level_set.gather_evaluate(level_set_as_heaviside, false, true);

              surface_tension.reinit(cell);

              curvature.reinit(cell);
              curvature.read_dof_values_plain(solution_curvature);
              curvature.evaluate(true, false);

              temperature_val.reinit(cell);
              temperature_val.read_dof_values_plain(temperature);
              temperature_val.evaluate(true, true);

              for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
                {
                  const auto n      = level_set.get_gradient(q_index);
                  const auto T      = temperature_val.get_value(q_index);
                  const auto grad_T = temperature_val.get_gradient(q_index);

                  Tensor<1, dim, VectorizedArray<double>> temp_surf_ten;

                  for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = 0; j < dim; ++j)
                      temp_surf_ten[i] = (i == j) ?
                                           -(make_vectorized_array<double>(1.) - n[i] * n[j]) *
                                             d_alpha0 * grad_T[j] :
                                           (n[i] * n[j]) * d_alpha0 * grad_T[j];

                  const auto alpha = compare_and_apply_mask<SIMDComparison::less_than>(
                    T,
                    T0,
                    VectorizedArray<double>(alpha0),
                    VectorizedArray<double>(alpha0) - VectorizedArray<double>(d_alpha0) * (T - T0));

                  for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
                    Assert(alpha[v] >= 0.0,
                           ExcMessage(
                             "The surface tension coefficient tends to be negative in "
                             "some regions. Check the value of the temperature dependent surface "
                             "tension coefficient."));

                  surface_tension.submit_value(alpha * n * curvature.get_value(q_index) +
                                                 temp_surf_ten,
                                               q_index);
                }
              surface_tension.integrate_scatter(true, false, force_rhs);
            }
        },
        force_rhs,
        level_set_as_heaviside,
        zero_out);

      solution_curvature.zero_out_ghosts();
    }
  };
} // namespace MeltPoolDG::Flow
