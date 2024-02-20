#include <deal.II/base/exceptions.h>

#include <meltpooldg/radiative_transport/pseudo_rte_operator.hpp>
#include <meltpooldg/radiative_transport/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <algorithm>


namespace MeltPoolDG::RadiativeTransport
{
  template <int dim, typename number>
  PseudoRTEOperator<dim, number>::PseudoRTEOperator(
    const ScratchData<dim>               &scratch_data_in,
    const RadiativeTransportData<double> &rte_data_in,
    const Tensor<1, dim, number>         &laser_direction_in,
    const VectorType                     &heaviside_in,
    const unsigned int                    rte_dof_idx_in,
    const unsigned int                    rte_quad_idx_in,
    const unsigned int                    hs_dof_idx_in)
    : scratch_data(scratch_data_in)
    , rte_data(rte_data_in)
    , laser_direction(laser_direction_in)
    , heaviside(heaviside_in)
    , rte_dof_idx(rte_dof_idx_in)
    , rte_quad_idx(rte_quad_idx_in)
    , hs_dof_idx(hs_dof_idx_in)
    , pure_gas_level_set(rte_data.absorptivity_gradient_based_data.avoid_div_zero_constant)
    , pure_liquid_level_set(1. - rte_data.absorptivity_gradient_based_data.avoid_div_zero_constant)
  {
    this->reset_dof_index(rte_dof_idx_in);
  }

  template <int dim, typename number>
  void
  PseudoRTEOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> intensity_vals(matrix_free, this->dof_idx, rte_quad_idx);
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            intensity_vals.reinit(cell);
            intensity_vals.read_dof_values(src);

            tangent_local_cell_operation(intensity_vals);

            intensity_vals.distribute_local_to_global(dst);
          }
      },
      dst,
      src,
      true /*zero out dst*/);
  }

  template <int dim, typename number>
  void
  PseudoRTEOperator<dim, number>::create_rhs(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number> last_intensity_vals(matrix_free,
                                                             this->dof_idx,
                                                             rte_quad_idx);
        FECellIntegrator<dim, 1, number> heaviside_vals(scratch_data.get_matrix_free(),
                                                        hs_dof_idx,
                                                        rte_quad_idx);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            last_intensity_vals.reinit(cell);
            last_intensity_vals.read_dof_values_plain(src);
            last_intensity_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients |
                                         EvaluationFlags::hessians);

            heaviside_vals.reinit(cell);
            heaviside_vals.read_dof_values_plain(heaviside);
            heaviside_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            for (unsigned int q_index = 0; q_index < last_intensity_vals.n_q_points; ++q_index)
              {
                const scalar last_I      = last_intensity_vals.get_value(q_index);
                const vector last_grad_I = last_intensity_vals.get_gradient(q_index);
                const scalar H           = heaviside_vals.get_value(q_index);
                const vector grad_H      = heaviside_vals.get_gradient(q_index);

                const scalar mu_A = compare_and_apply_mask<SIMDComparison::greater_than>(
                  compute_invalid_mask(last_I, last_grad_I, H, pure_liquid_level_set),
                  VectorizedArray<double>(1e-16),
                  /*true*/ VectorizedArray<double>(rte_data.avoid_singular_matrix_absorptivity),
                  /*false*/
                  compute_mu(rte_data,
                             H,
                             grad_H,
                             laser_direction,
                             rte_data.absorptivity_gradient_based_data.avoid_div_zero_constant));

                scalar advec_term = -rte_data.pseudo_time_stepping.advection_term_scaling *
                                    scalar_product(laser_direction, last_grad_I);
                scalar diffu_term = rte_data.pseudo_time_stepping.diffusion_term_scaling *
                                    last_intensity_vals.get_laplacian(q_index);
                scalar scalar_submission = advec_term - mu_A * last_I + diffu_term;

                last_intensity_vals.submit_value(last_I + this->time_increment * scalar_submission,
                                                 q_index);
              }
            last_intensity_vals.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      dst,
      src,
      false /*don't zero out to consider inhomogeneous Dirichlet BC*/);
  }

  template <int dim, typename number>
  void
  PseudoRTEOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    // note: not thread safe!!!
    const auto &matrix_free = scratch_data.get_matrix_free();

    // compute matrix (only cell contributions)
    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      scratch_data.get_constraint(this->dof_idx),
      system_matrix,
      [&](auto &intensity_vals) { tangent_local_cell_operation(intensity_vals); },
      this->dof_idx, // absorptivity information closes the heaviside function definition
      rte_quad_idx);

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  PseudoRTEOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, rte_dof_idx);
    // note: not thread safe!!!
    const auto &matrix_free = scratch_data.get_matrix_free();

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute diagonal ...
    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal,
      [&](auto &intensity_vals) {
        const unsigned int current_cell_index = intensity_vals.get_current_cell_index();

        tangent_local_cell_operation(intensity_vals);

        old_cell_index = current_cell_index;
      },
      rte_dof_idx,
      rte_quad_idx);

    // ... and invert it
    const double linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  PseudoRTEOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number> &intensity_vals) const
  {
    intensity_vals.evaluate(EvaluationFlags::values);
    for (unsigned int q_index = 0; q_index < intensity_vals.n_q_points; ++q_index)
      {
        intensity_vals.submit_value(intensity_vals.get_value(q_index), q_index);
      }
    intensity_vals.integrate(EvaluationFlags::values);
  }

  template class PseudoRTEOperator<1, double>;
  template class PseudoRTEOperator<2, double>;
  template class PseudoRTEOperator<3, double>;
} // namespace MeltPoolDG::RadiativeTransport
