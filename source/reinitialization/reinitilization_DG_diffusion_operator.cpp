#include <meltpooldg/reinitialization/reinitilization_DG_diffusion_operator.hpp>


namespace MeltPoolDG::LevelSet
{

  template <int dim, typename Number>
  RIDGDiffusionOperator<dim, Number>::RIDGDiffusionOperator(
    const MeltPoolDG::ScratchData<dim> &scratch_datain,
    const ReinitializationData<Number> &reinit_data_in,
    const unsigned int                  reinit_dof_idx_in,
    const unsigned int                  reinit_quad_idx_in)
    : scratch_data(scratch_datain)
    , reinit_data(reinit_data_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
  {}

  template <int dim, typename Number>
  void
  RIDGDiffusionOperator<dim, Number>::compute_viscosity_value()
  {
    // The value for the artificial viscosity is determined by the smallest enabled element size.
    viscosity = reinit_data.factor_diffusivity * scratch_data.get_min_cell_size() /
                ((Number)scratch_data.get_degree(reinit_dof_idx));
  }

  template <int dim, typename Number>
  void
  RIDGDiffusionOperator<dim, Number>::compute_penalty_parameter()
  {
    const Number fe_degree = (Number)scratch_data.get_degree(reinit_dof_idx);
    // Resize
    const unsigned int n_macro_cells = scratch_data.get_matrix_free().n_cell_batches() +
                                       scratch_data.get_matrix_free().n_ghost_cell_batches();
    array_penalty_parameter.resize(n_macro_cells);
    for (uint macro_cells = 0; macro_cells < n_macro_cells; ++macro_cells)
      {
        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled =
          scratch_data.get_matrix_free().n_active_entries_per_cell_batch(macro_cells);
        for (uint lane = 0; lane < n_lanes_filled; ++lane)
          {
            auto cell = scratch_data.get_matrix_free().get_cell_iterator(macro_cells, lane);
            array_penalty_parameter[macro_cells][lane] = 1. / cell->minimum_vertex_distance() *
                                                         (fe_degree + 1.0) * (fe_degree + 1.0) *
                                                         reinit_data.IP_diffusion;
          }
      }
  }

  template <int dim, typename Number>
  void
  RIDGDiffusionOperator<dim, Number>::local_apply_domain_RI_diffusion(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, Number> eval(data, reinit_dof_idx, reinit_quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto flux = viscosity * eval.get_gradient(q);
            eval.submit_gradient(-flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename Number>
  void
  RIDGDiffusionOperator<dim, Number>::local_apply_inner_face_RI_diffusion(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, Number> eval_minus(data, true, reinit_dof_idx, reinit_quad_idx);

    FEFaceIntegrator<dim, 1, Number> eval_plus(data, false, reinit_dof_idx, reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        eval_plus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const auto sigmaF = std::max(eval_minus.read_cell_data(array_penalty_parameter),
                                     eval_plus.read_cell_data(array_penalty_parameter));

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            // 1st face integral
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = eval_plus.get_value(q);

            const auto flux_1 = 0.5 * (u_minus - u_plus) * viscosity;

            eval_minus.submit_normal_derivative(flux_1, q);
            eval_plus.submit_normal_derivative(flux_1, q);

            // 2nd and 3rd (=penalty) face integral
            const auto u_minus_normal_grad = eval_minus.get_normal_derivative(q);
            const auto u_plus_normal_grad  = eval_plus.get_normal_derivative(q);

            const auto flux_2 = 0.5 * (u_minus_normal_grad + u_plus_normal_grad) * viscosity -
                                (u_minus - u_plus) * viscosity * sigmaF;

            eval_minus.submit_value(flux_2, q);
            eval_plus.submit_value(-flux_2, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename Number>
  void
  RIDGDiffusionOperator<dim, Number>::local_apply_boundary_face_RI_diffusion(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, Number> eval_minus(data, true, reinit_dof_idx, reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const auto sigmaF = eval_minus.read_cell_data(array_penalty_parameter);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            // 1st face integral
            const auto u_minus = eval_minus.get_value(q);

            const auto u_plus = u_minus;
            const auto flux_1 = 0.5 * (u_minus - u_plus) * viscosity;

            eval_minus.submit_normal_derivative(flux_1, q);

            // 2nd and 3rd (=penalty) face integral
            const auto u_minus_normal_grad = eval_minus.get_normal_derivative(q);

            // Assume same gradients.
            const auto u_plus_normal_grad = -u_minus_normal_grad;

            const auto flux_2 = 0.5 * (u_minus_normal_grad + u_plus_normal_grad) * viscosity -
                                (u_minus - u_plus) * viscosity * sigmaF;

            eval_minus.submit_value(flux_2, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }


  template <int dim, typename Number>
  double
  RIDGDiffusionOperator<dim, Number>::get_viscosity() const
  {
    return viscosity;
  }

  template <int dim, typename Number>
  void
  RIDGDiffusionOperator<dim, Number>::apply_operator([[maybe_unused]] const Number time,
                                                     VectorType                   &dst,
                                                     const VectorType             &src) const
  {
    scratch_data.get_matrix_free().loop(
      &RIDGDiffusionOperator<dim, Number>::local_apply_domain_RI_diffusion,
      &RIDGDiffusionOperator<dim, Number>::local_apply_inner_face_RI_diffusion,
      &RIDGDiffusionOperator<dim, Number>::local_apply_boundary_face_RI_diffusion,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
      MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }

  template class RIDGDiffusionOperator<1>;
  template class RIDGDiffusionOperator<2>;
  template class RIDGDiffusionOperator<3>;
} // namespace MeltPoolDG::LevelSet