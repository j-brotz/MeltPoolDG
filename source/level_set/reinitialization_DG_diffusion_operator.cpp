#include <meltpooldg/level_set/reinitialization_DG_diffusion_operator.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>



namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationDGDiffusionOperator<dim, number>::ReinitializationDGDiffusionOperator(
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_datain,
    const ReinitializationData<number>              &reinit_data_in,
    const unsigned int                               reinit_dof_idx_in,
    const unsigned int                               reinit_quad_idx_in,
    const VectorType                                &curvature_in,
    const BlockVectorType                           &normal_vector_in,
    const VectorType                                &smooth_signum_in)
    : scratch_data(scratch_datain)
    , reinit_data(reinit_data_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , curvature(curvature_in)
    , normal_vector(normal_vector_in)
    , smooth_signum(smooth_signum_in)
  {}

  template <int dim, typename number>
  void
  ReinitializationDGDiffusionOperator<dim, number>::compute_diffusitivity_value()
  {
    scratch_data.initialize_dof_vector(diffusitivity, reinit_dof_idx);

    const auto &data = scratch_data.get_matrix_free();

    {
      FECellIntegrator<dim, 1, number> eval_diffusitivity(data, reinit_dof_idx, reinit_quad_idx);
      FECellIntegrator<dim, 1, number> eval_smooth_signum(data, reinit_dof_idx, reinit_quad_idx);

      auto const diffusion_const_factor =
        scratch_data.get_min_cell_size() /
        ((static_cast<number>(scratch_data.get_degree(reinit_dof_idx))));

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          eval_diffusitivity.reinit(cell);
          eval_smooth_signum.reinit(cell);

          eval_smooth_signum.read_dof_values(smooth_signum);

          for (unsigned int q = 0; q < eval_diffusitivity.dofs_per_cell; ++q)
            {
              if (reinit_data.reinitilization_DG_specific_data.use_spatially_constant_diffusion)
                {
                  /**
                   * The value for the artificial diffusitivity is determined by the smallest
                   * enabled element size.
                   */
                  const VectorizedArray<number> diffusitivity_value =
                    reinit_data.reinitilization_DG_specific_data.factor_diffusivity *
                    diffusion_const_factor;
                  eval_diffusitivity.submit_dof_value(diffusitivity_value, q);
                }
              else
                {
                  /**
                   * Local diffusitivity value is determined from a local analysis of the
                   * characterisc speed
                   */
                  const auto diffusitivity_value =
                    std::abs(eval_smooth_signum.get_dof_value(q)) * diffusion_const_factor;
                  eval_diffusitivity.submit_dof_value(diffusitivity_value, q);
                }
            }

          eval_diffusitivity.set_dof_values(diffusitivity);
        }

      if (!diffusitivity.has_ghost_elements())
        diffusitivity.update_ghost_values();
    }
  }

  template <int dim, typename number>
  void
  ReinitializationDGDiffusionOperator<dim, number>::compute_penalty_parameter()
  {
    const number fe_degree = (static_cast<number>(scratch_data.get_degree(reinit_dof_idx)));
    // Resize
    const unsigned int n_macro_cells = scratch_data.get_matrix_free().n_cell_batches() +
                                       scratch_data.get_matrix_free().n_ghost_cell_batches();
    array_penalty_parameter.resize(n_macro_cells);
    for (unsigned int macro_cells = 0; macro_cells < n_macro_cells; ++macro_cells)
      {
        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled =
          scratch_data.get_matrix_free().n_active_entries_per_cell_batch(macro_cells);
        for (unsigned int lane = 0; lane < n_lanes_filled; ++lane)
          {
            auto cell = scratch_data.get_matrix_free().get_cell_iterator(macro_cells, lane);
            array_penalty_parameter[macro_cells][lane] =
              1. / cell->minimum_vertex_distance() * (fe_degree + 1.0) * (fe_degree + 1.0) *
              reinit_data.reinitilization_DG_specific_data.IP_diffusion;
          }
      }
  }

  template <int dim, typename number>
  void
  ReinitializationDGDiffusionOperator<dim, number>::local_apply_domain(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, number>   eval(data, reinit_dof_idx, reinit_quad_idx);
    FECellIntegrator<dim, 1, number>   eval_curvature(data, reinit_dof_idx, reinit_quad_idx);
    FECellIntegrator<dim, dim, number> eval_normal(data, reinit_dof_idx, reinit_quad_idx);
    FECellIntegrator<dim, 1, number>   eval_diffusitivity(data, reinit_dof_idx, reinit_quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval_diffusitivity.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::gradients);
        eval_diffusitivity.gather_evaluate(diffusitivity,
                                           EvaluationFlags::values | EvaluationFlags::gradients);

        if (reinit_data.reinitilization_DG_specific_data.use_directed_diffusion_stabilization)
          {
            eval_curvature.reinit(cell);
            eval_normal.reinit(cell);

            eval_curvature.gather_evaluate(curvature, EvaluationFlags::values);
            eval_normal.read_dof_values_plain(normal_vector);
            eval_normal.evaluate(EvaluationFlags::values);
          }

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u_gradient             = eval.get_gradient(q);
            const auto diffusitivity_value    = eval_diffusitivity.get_value(q);
            const auto diffusitivity_gradient = eval_diffusitivity.get_gradient(q);

            const auto flux = diffusitivity_value * u_gradient;

            if (reinit_data.reinitilization_DG_specific_data.use_directed_diffusion_stabilization)
              {
                const Tensor<1, dim, VectorizedArray<number>> n_phi =
                  MeltPoolDG::VectorTools::normalize<dim>(eval_normal.get_value(q), 1.0e-16);
                auto const directed_diffusion_flux =
                  (diffusitivity_gradient * n_phi -
                   eval_curvature.get_value(q) * diffusitivity_value) *
                  scalar_product(u_gradient, n_phi);
                eval.submit_value(-directed_diffusion_flux, q);
              }

            eval.submit_gradient(-flux, q);
          }
        if (reinit_data.reinitilization_DG_specific_data.use_directed_diffusion_stabilization)
          {
            eval.integrate_scatter(EvaluationFlags::gradients | EvaluationFlags::values, dst);
          }
        else
          {
            eval.integrate_scatter(EvaluationFlags::gradients, dst);
          }
      }
  }

  template <int dim, typename number>
  void
  ReinitializationDGDiffusionOperator<dim, number>::local_apply_inner_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, number> eval_minus(data, true, reinit_dof_idx, reinit_quad_idx);
    FEFaceIntegrator<dim, 1, number> eval_plus(data, false, reinit_dof_idx, reinit_quad_idx);
    FEFaceIntegrator<dim, 1, number> eval_diffusitivity_minus(data,
                                                              true,
                                                              reinit_dof_idx,
                                                              reinit_quad_idx);
    FEFaceIntegrator<dim, 1, number> eval_diffusitivity_plus(data,
                                                             false,
                                                             reinit_dof_idx,
                                                             reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);
        eval_diffusitivity_minus.reinit(face);
        eval_diffusitivity_plus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        eval_plus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        eval_diffusitivity_minus.gather_evaluate(diffusitivity, EvaluationFlags::values);
        eval_diffusitivity_plus.gather_evaluate(diffusitivity, EvaluationFlags::values);

        const auto sigmaF = std::max(eval_minus.read_cell_data(array_penalty_parameter),
                                     eval_plus.read_cell_data(array_penalty_parameter));

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            // 1st face integral
            const auto u_minus             = eval_minus.get_value(q);
            const auto u_plus              = eval_plus.get_value(q);
            const auto diffusitivity_minus = eval_diffusitivity_minus.get_value(q);
            const auto diffusitivity_plus  = eval_diffusitivity_plus.get_value(q);

            const auto u_minus_normal_grad = eval_minus.get_normal_derivative(q);
            const auto u_plus_normal_grad  = eval_plus.get_normal_derivative(q);

            const auto average   = 0.5 * (diffusitivity_minus * u_minus_normal_grad +
                                        diffusitivity_plus * u_plus_normal_grad);
            const auto SIPG_term = 0.5 * (u_minus - u_plus);


            eval_minus.submit_normal_derivative(diffusitivity_minus * SIPG_term, q);
            eval_plus.submit_normal_derivative(diffusitivity_plus * SIPG_term, q);

            eval_minus.submit_value(average - (u_minus - u_plus) * sigmaF, q);
            eval_plus.submit_value(-average + (u_minus - u_plus) * sigmaF, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  ReinitializationDGDiffusionOperator<dim, number>::local_apply_boundary_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, number> eval_minus(data, true, reinit_dof_idx, reinit_quad_idx);
    FEFaceIntegrator<dim, 1, number> eval_diffusitivity_minus(data,
                                                              true,
                                                              reinit_dof_idx,
                                                              reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_diffusitivity_minus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        eval_diffusitivity_minus.gather_evaluate(diffusitivity, EvaluationFlags::values);

        const auto sigmaF = eval_minus.read_cell_data(array_penalty_parameter);
        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            // 1st face integral
            const auto u_minus             = eval_minus.get_value(q);
            const auto u_plus              = u_minus;
            const auto diffusitivity_minus = eval_diffusitivity_minus.get_value(q);
            const auto diffusitivity_plus  = diffusitivity_minus;

            const auto u_minus_normal_grad = eval_minus.get_normal_derivative(q);
            const auto u_plus_normal_grad  = u_minus_normal_grad;

            const auto average   = 0.5 * (diffusitivity_minus * u_minus_normal_grad +
                                        diffusitivity_plus * u_plus_normal_grad);
            const auto SIPG_term = 0.5 * (u_minus - u_plus);


            eval_minus.submit_normal_derivative(diffusitivity_minus * SIPG_term, q);

            eval_minus.submit_value(average - (u_minus - u_plus) * sigmaF, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  number
  ReinitializationDGDiffusionOperator<dim, number>::get_max_diffusitivity() const
  {
    return diffusitivity.linfty_norm();
  }

  template <int dim, typename number>
  void
  ReinitializationDGDiffusionOperator<dim, number>::apply_operator(
    [[maybe_unused]] const number                          time,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    scratch_data.get_matrix_free().loop(
      &ReinitializationDGDiffusionOperator<dim, number>::local_apply_domain,
      &ReinitializationDGDiffusionOperator<dim, number>::local_apply_inner_face,
      &ReinitializationDGDiffusionOperator<dim, number>::local_apply_boundary_face,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::unspecified,
      MatrixFree<dim, number>::DataAccessOnFaces::unspecified);

    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type inverse =
      [dof_idx  = reinit_dof_idx,
       quad_idx = reinit_quad_idx](const MatrixFree<dim, number>                    &matrix_free,
                                   LinearAlgebra::distributed::Vector<number>       &dst,
                                   const LinearAlgebra::distributed::Vector<number> &src,
                                   const std::pair<unsigned int, unsigned int>       cell_range) {
        Utilities::MatrixFree::local_apply_inverse_mass_matrix<dim, 1, number>(
          matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };

    scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }

  template class ReinitializationDGDiffusionOperator<1, double>;
  template class ReinitializationDGDiffusionOperator<2, double>;
  template class ReinitializationDGDiffusionOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
