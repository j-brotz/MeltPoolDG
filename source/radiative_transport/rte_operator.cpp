#include <deal.II/base/exceptions.h>

#include <meltpooldg/radiative_transport/rte_operator.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::RadiativeTransport
{
  template <int dim, typename number>
  RadiativeTransportOperator<dim, number>::RadiativeTransportOperator(
    const ScratchData<dim> &              scratch_data_in,
    const RadiativeTransportData<double> &rte_data_in,
    VectorType &                          intensity_in,
    const VectorType &                    heaviside_in,
    const unsigned int                    rte_dof_idx_in,
    const unsigned int                    rte_quad_idx_in,
    const unsigned int                    hs_dof_idx_in)
    : scratch_data(scratch_data_in)
    , rte_data(rte_data_in)
    , intensity(intensity_in)
    , heaviside(heaviside_in)
    , rte_dof_idx(rte_dof_idx_in)
    , rte_quad_idx(rte_quad_idx_in)
    , hs_dof_idx(hs_dof_idx_in)
  {}

  template <int dim, typename number>
  void
  RadiativeTransportOperator<dim, number>::assemble_matrixbased(const VectorType &intensity_old,
                                                                SparseMatrixType &matrix,
                                                                VectorType &      rhs) const
  {
    (void)intensity_old;
    (void)matrix;
    (void)rhs;
    AssertThrow(false, ExcNotImplemented());
  }

  template <int dim, typename number>
  void
  RadiativeTransportOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> intensity_vals(matrix_free, rte_dof_idx, rte_quad_idx);
        FECellIntegrator<dim, 1, number> heaviside_vals(scratch_data.get_matrix_free(),
                                                        hs_dof_idx,
                                                        rte_quad_idx);
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            intensity_vals.reinit(cell);
            intensity_vals.read_dof_values(src);

            tangent_local_cell_operation(intensity_vals, heaviside_vals, true);

            intensity_vals.distribute_local_to_global(dst);
          }
      },
      dst,
      src,
      true /*zero out dst*/);
  }

  template <int dim, typename number>
  void
  RadiativeTransportOperator<dim, number>::create_rhs(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number> intensity_vals(matrix_free, rte_dof_idx, rte_quad_idx);
        FECellIntegrator<dim, 1, number> heaviside_vals(scratch_data.get_matrix_free(),
                                                        hs_dof_idx,
                                                        rte_quad_idx);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            intensity_vals.reinit(cell);
            intensity_vals.read_dof_values_plain(src);

            heaviside_vals.reinit(cell);
            heaviside_vals.read_dof_values_plain(heaviside);
            heaviside_vals.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < intensity_vals.n_q_points; ++q_index)
              {
                AssertThrow(false, ExcNotImplemented());
                intensity_vals.submit_value(0.0 /*TODO*/, q_index);
              }
            intensity_vals.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      dst,
      src,
      true /*zero out dst*/);
  }

  template <int dim, typename number>
  void
  RadiativeTransportOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    heaviside.update_ghost_values();

    // note: not thread safe!!!
    const auto &                     matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, 1, number> heaviside_vals(matrix_free, hs_dof_idx, rte_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute matrix (only cell contributions)
    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      scratch_data.get_constraint(rte_dof_idx),
      system_matrix,
      [&](auto &intensity_vals) {
        const unsigned int current_cell_index = intensity_vals.get_current_cell_index();

        tangent_local_cell_operation(intensity_vals,
                                     heaviside_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      rte_dof_idx,
      rte_quad_idx);

    system_matrix.compress(VectorOperation::add);

    heaviside.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  RadiativeTransportOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, rte_dof_idx);
    // note: not thread safe!!!
    const auto &                     matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, 1, number> heaviside_vals(matrix_free, hs_dof_idx, rte_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    heaviside.update_ghost_values();

    // compute diagonal ...
    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal,
      [&](auto &intensity_vals) {
        const unsigned int current_cell_index = intensity_vals.get_current_cell_index();

        tangent_local_cell_operation(intensity_vals,
                                     heaviside_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      rte_dof_idx,
      rte_quad_idx);

    heaviside.zero_out_ghost_values();

    // ... and invert it
    const double linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  RadiativeTransportOperator<dim, number>::reinit()
  {
    return;
  }

  template <int dim, typename number>
  void
  RadiativeTransportOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number> &intensity_vals,
    FECellIntegrator<dim, 1, number> &heaviside_vals,
    const bool                        do_reinit_cells) const
  {
    (void)heaviside_vals;
    (void)do_reinit_cells;

    intensity_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q_index = 0; q_index < intensity_vals.n_q_points; ++q_index)
      {
        AssertThrow(false, ExcNotImplemented());
        intensity_vals.submit_value(0.0 /*TODO*/, q_index);
        intensity_vals.submit_gradient(vector() /*TODO*/, q_index);
      }

    intensity_vals.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template class RadiativeTransportOperator<1, double>;
  template class RadiativeTransportOperator<2, double>;
  template class RadiativeTransportOperator<3, double>;
} // namespace MeltPoolDG::RadiativeTransport
