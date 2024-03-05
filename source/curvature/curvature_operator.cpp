#include <meltpooldg/curvature/curvature_operator.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  CurvatureOperator<dim, number>::CurvatureOperator(const ScratchData<dim>      &scratch_data_in,
                                                    const CurvatureData<double> &curvature_data_in,
                                                    const unsigned int           curv_dof_idx_in,
                                                    const unsigned int           curv_quad_idx_in,
                                                    const unsigned int           normal_dof_idx_in,
                                                    const unsigned int           ls_dof_idx_in,
                                                    const VectorType *solution_level_set_in)
    : scratch_data(scratch_data_in)
    , curvature_data(curvature_data_in)
    , curv_dof_idx(curv_dof_idx_in)
    , curv_quad_idx(curv_quad_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
    , tolerance_normal_vector(
        UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                              scratch_data.get_mapping()))
    , ls_dof_idx(ls_dof_idx_in)
    , solution_level_set(solution_level_set_in)

  {
    this->reset_dof_index(curv_dof_idx_in);

    AssertThrow(!curvature_data.narrow_band.enable || solution_level_set,
                ExcMessage(
                  "Level set solution vector must not be nullptr for a narrow band computation."));
  }

  template <int dim, typename number>
  void
  CurvatureOperator<dim, number>::assemble_matrixbased(
    const CurvatureOperator::BlockVectorType &solution_normal_vector_in,
    CurvatureOperator::SparseMatrixType      &matrix,
    VectorType                               &rhs) const
  {
    BlockVectorType unit_normal(solution_normal_vector_in);

    FEValues<dim> curv_values(scratch_data.get_mapping(),
                              scratch_data.get_dof_handler(curv_dof_idx).get_fe(),
                              scratch_data.get_quadrature(curv_quad_idx),
                              update_values | update_gradients | update_quadrature_points |
                                update_JxW_values);

    FEValues<dim> normal_values(scratch_data.get_mapping(),
                                scratch_data.get_dof_handler(normal_dof_idx).get_fe(),
                                scratch_data.get_quadrature(curv_quad_idx),
                                update_gradients);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(curv_dof_idx);

    FullMatrix<double>                   curvature_cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                       curvature_cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const unsigned int n_q_points = curv_values.get_quadrature().size();

    matrix = 0.0;
    rhs    = 0.0;

    for (const auto &cell : scratch_data.get_dof_handler(curv_dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          curv_values.reinit(cell);
          cell->get_dof_indices(local_dof_indices);

          curvature_cell_matrix = 0.0;
          curvature_cell_rhs    = 0.0;

          const double damping = compute_cell_size_dependent_filter_parameter<dim>(
            scratch_data, curv_dof_idx, cell, curvature_data.filter_parameter);

          // set unit normal vector at DoFs to compute the RHS of the curvature operator
          //
          // (N, -∇*n)
          //
          // 1) gather (componentwise)
          std::vector<Vector<double>> normal_vector_at_cell(dim, Vector<double>(dofs_per_cell));
          for (unsigned int d = 0; d < dim; ++d)
            cell->get_dof_values(solution_normal_vector_in.block(d), normal_vector_at_cell[d]);

          std::vector<Tensor<1, dim>> unit_normal_at_cell(dofs_per_cell, Tensor<1, dim>());
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            for (unsigned int d = 0; d < dim; ++d)
              unit_normal_at_cell[k][d] = normal_vector_at_cell[d][k];

          // 2) normalize
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              const double n_norm = unit_normal_at_cell[k].norm();
              if (n_norm > tolerance_normal_vector)
                unit_normal_at_cell[k] /= n_norm;
              else
                unit_normal_at_cell[k] = 0.0;
            }

          // 3) scatter (componentwise)
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            for (unsigned int d = 0; d < dim; ++d)
              normal_vector_at_cell[d][k] = unit_normal_at_cell[k][d];

          for (unsigned int d = 0; d < dim; ++d)
            cell->set_dof_values(normal_vector_at_cell[d], unit_normal.block(d));

          // 4) evaluate divergence at quadrature points
          normal_values.reinit(cell);
          std::vector<double> div_n_at_q(n_q_points);

          for (unsigned int d = 0; d < dim; ++d)
            {
              std::vector<Tensor<1, dim>> temp(n_q_points, Tensor<1, dim>());
              normal_values.get_function_gradients(unit_normal.block(d), temp);
              for (unsigned int q = 0; q < n_q_points; ++q)
                div_n_at_q[q] += temp[q][d];
            }

          for (const unsigned int q_index : curv_values.quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const double         phi_i      = curv_values.shape_value(i, q_index);
                  const Tensor<1, dim> grad_phi_i = curv_values.shape_grad(i, q_index);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      const double         phi_j      = curv_values.shape_value(j, q_index);
                      const Tensor<1, dim> grad_phi_j = curv_values.shape_grad(j, q_index);

                      curvature_cell_matrix(i, j) +=
                        (phi_i * phi_j + damping * grad_phi_i * grad_phi_j) *
                        curv_values.JxW(q_index);
                    }
                  curvature_cell_rhs(i) -= (phi_i * div_n_at_q[q_index] * curv_values.JxW(q_index));
                }
            }

          // assembly
          cell->get_dof_indices(local_dof_indices);
          scratch_data.get_constraint(curv_dof_idx)
            .distribute_local_to_global(
              curvature_cell_matrix, curvature_cell_rhs, local_dof_indices, matrix, rhs);

        } // end of cell loop
    matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  CurvatureOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> curv_vals(matrix_free, curv_dof_idx, curv_quad_idx);
        FECellIntegrator<dim, 1, number> level_set_vals(scratch_data.get_matrix_free(),
                                                        ls_dof_idx,
                                                        curv_quad_idx);
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            curv_vals.reinit(cell);
            curv_vals.read_dof_values(src);

            tangent_local_cell_operation(curv_vals, level_set_vals, true);

            curv_vals.distribute_local_to_global(dst);
          }
      },
      dst,
      src,
      true /*zero out dst*/);
  }

  template <int dim, typename number>
  void
  CurvatureOperator<dim, number>::create_rhs(VectorType                               &dst,
                                             const CurvatureOperator::BlockVectorType &src) const
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, BlockVectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number>   curvature(matrix_free, curv_dof_idx, curv_quad_idx);
        FECellIntegrator<dim, dim, number> normal_vector(matrix_free,
                                                         normal_dof_idx,
                                                         curv_quad_idx);
        FECellIntegrator<dim, 1, number>   level_set(scratch_data.get_matrix_free(),
                                                   ls_dof_idx,
                                                   curv_quad_idx);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            curvature.reinit(cell);

            normal_vector.reinit(cell);
            normal_vector.read_dof_values_plain(src);
            normal_vector.evaluate(EvaluationFlags::values);

            if (curvature_data.narrow_band.enable)
              {
                level_set.reinit(cell);
                level_set.read_dof_values_plain(*solution_level_set);
                level_set.evaluate(EvaluationFlags::values);
              }

            // submit unit normal vector as DoF value
            for (unsigned int i = 0; i < normal_vector.dofs_per_component; ++i)
              {
                // We need to distinguish dim==1 and dim>1 since two different data types
                // need to be submitted.
                if constexpr (dim > 1)
                  {
                    const Tensor<1, dim, VectorizedArray<double>> n_phi =
                      MeltPoolDG::VectorTools::normalize<dim>(normal_vector.get_dof_value(i),
                                                              tolerance_normal_vector);
                    normal_vector.submit_dof_value(n_phi, i);
                  }
                else
                  {
                    const VectorizedArray<double> n_phi =
                      compare_and_apply_mask<SIMDComparison::greater_than>(
                        normal_vector.get_dof_value(i), tolerance_normal_vector, 1.0, -1.0);

                    normal_vector.submit_dof_value(n_phi, i);
                  }
              }

            normal_vector.evaluate(EvaluationFlags::gradients);

            for (unsigned int q_index = 0; q_index < curvature.n_q_points; ++q_index)
              {
                const VectorizedArray<number> narrow_band_mask =
                  (curvature_data.narrow_band.enable) ?
                    VectorTools::compute_mask_narrow_band<dim>(
                      level_set.get_value(q_index),
                      curvature_data.narrow_band.level_set_threshold) :
                    1.0;
                curvature.submit_value(-narrow_band_mask * normal_vector.get_divergence(q_index),
                                       q_index);
              }
            curvature.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      dst,
      src,
      true /*zero out dst*/);
  }

  template <int dim, typename number>
  void
  CurvatureOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    if (solution_level_set)
      solution_level_set->update_ghost_values();

    // note: not thread safe!!!
    const auto                      &matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, 1, number> level_set_vals(matrix_free, ls_dof_idx, curv_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute matrix (only cell contributions)
    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      scratch_data.get_constraint(curv_dof_idx),
      system_matrix,
      [&](auto &curv_vals) {
        const unsigned int current_cell_index = curv_vals.get_current_cell_index();

        tangent_local_cell_operation(curv_vals,
                                     level_set_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      curv_dof_idx,
      curv_quad_idx);

    if (solution_level_set)
      solution_level_set->zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  CurvatureOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, curv_dof_idx);
    // note: not thread safe!!!
    const auto                      &matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, 1, number> level_set_vals(matrix_free, ls_dof_idx, curv_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    if (solution_level_set)
      solution_level_set->update_ghost_values();

    // compute diagonal ...
    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal,
      [&](auto &curv_vals) {
        const unsigned int current_cell_index = curv_vals.get_current_cell_index();

        tangent_local_cell_operation(curv_vals,
                                     level_set_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      curv_dof_idx,
      curv_quad_idx);

    if (solution_level_set)
      solution_level_set->zero_out_ghost_values();

    // ... and invert it
    const double linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  CurvatureOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number> &curv_vals,
    FECellIntegrator<dim, 1, number> &level_set_vals,
    const bool                        do_reinit_cells) const
  {
    curv_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (do_reinit_cells)
      {
        if (curvature_data.narrow_band.enable)
          {
            level_set_vals.reinit(curv_vals.get_current_cell_index());
            level_set_vals.read_dof_values_plain(*solution_level_set);
            level_set_vals.evaluate(EvaluationFlags::values);
          }
      }

    for (unsigned int q_index = 0; q_index < curv_vals.n_q_points; ++q_index)
      {
        const VectorizedArray<number> narrow_band_mask =
          (curvature_data.narrow_band.enable) ?
            VectorTools::compute_mask_narrow_band<dim>(
              level_set_vals.get_value(q_index), curvature_data.narrow_band.level_set_threshold) :
            1.0;

        curv_vals.submit_value(narrow_band_mask * curv_vals.get_value(q_index), q_index);
        curv_vals.submit_gradient(narrow_band_mask * damping[curv_vals.get_current_cell_index()] *
                                    curv_vals.get_gradient(q_index),
                                  q_index);
      }

    curv_vals.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  CurvatureOperator<dim, number>::reinit()
  {
    if (curvature_data.linear_solver.do_matrix_free)
      {
        damping.resize_fast(scratch_data.get_matrix_free().n_cell_batches());

        for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
          {
            damping[cell] = compute_cell_size_dependent_filter_parameter_mf<dim>(
              scratch_data, curv_dof_idx, cell, curvature_data.filter_parameter);
          }
      }
  }


  template class CurvatureOperator<1, double>;
  template class CurvatureOperator<2, double>;
  template class CurvatureOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
