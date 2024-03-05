#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>


namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  NormalVectorOperator<dim, number>::NormalVectorOperator(
    const ScratchData<dim>         &scratch_data_in,
    const NormalVectorData<double> &normal_vector_data_in,
    const unsigned int              normal_dof_idx_in,
    const unsigned int              normal_quad_idx_in,
    const unsigned int              ls_dof_idx_in,
    const VectorType               *solution_level_set_in)
    : scratch_data(scratch_data_in)
    , normal_vector_data(normal_vector_data_in)
    , normal_dof_idx(normal_dof_idx_in)
    , normal_quad_idx(normal_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , solution_level_set(solution_level_set_in)
  {
    this->reset_dof_index(normal_dof_idx_in);
  }

  template <int dim, typename number>
  void
  NormalVectorOperator<dim, number>::assemble_matrixbased(const VectorType &level_set_in,
                                                          SparseMatrixType &matrix,
                                                          BlockVectorType  &rhs) const
  {
    FEValues<dim> normal_values(scratch_data.get_mapping(),
                                scratch_data.get_dof_handler(normal_dof_idx).get_fe(),
                                scratch_data.get_quadrature(normal_quad_idx),
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values);

    FEValues<dim> ls_values(scratch_data.get_mapping(),
                            scratch_data.get_dof_handler(ls_dof_idx).get_fe(),
                            scratch_data.get_quadrature(normal_quad_idx),
                            update_gradients);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(normal_dof_idx);

    FullMatrix<double>                   normal_cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<Vector<double>>          normal_cell_rhs(dim, Vector<double>(dofs_per_cell));
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const unsigned int n_q_points = normal_values.get_quadrature().size();

    std::vector<Tensor<1, dim>> normal_at_q(n_q_points, Tensor<1, dim>());

    matrix = 0.0;
    rhs    = 0.0;

    for (const auto &cell : scratch_data.get_dof_handler(normal_dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          normal_values.reinit(cell);
          ls_values.reinit(cell);
          cell->get_dof_indices(local_dof_indices);

          normal_cell_matrix = 0.0;
          for (auto &normal_cell : normal_cell_rhs)
            normal_cell = 0.0;

          ls_values.get_function_gradients(
            level_set_in, normal_at_q); // compute normals from level set solution at tau=0

          const double damping = compute_cell_size_dependent_filter_parameter<dim>(
            scratch_data, normal_dof_idx, cell, normal_vector_data.filter_parameter);

          for (const unsigned int q_index : normal_values.quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const double         phi_i      = normal_values.shape_value(i, q_index);
                  const Tensor<1, dim> grad_phi_i = normal_values.shape_grad(i, q_index);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      const double         phi_j      = normal_values.shape_value(j, q_index);
                      const Tensor<1, dim> grad_phi_j = normal_values.shape_grad(j, q_index);

                      //clang-format off
                      normal_cell_matrix(i, j) +=
                        (phi_i * phi_j + damping * grad_phi_i * grad_phi_j) *
                        normal_values.JxW(q_index);
                      //clang-format on
                    }

                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      //clang-format off
                      normal_cell_rhs[d](i) +=
                        phi_i * normal_at_q[q_index][d] * normal_values.JxW(q_index);
                      //clang-format on
                    }
                }
            }

          // assembly
          cell->get_dof_indices(local_dof_indices);

          scratch_data.get_constraint(normal_dof_idx)
            .distribute_local_to_global(normal_cell_matrix, local_dof_indices, matrix);
          for (unsigned int d = 0; d < dim; ++d)
            scratch_data.get_constraint(normal_dof_idx)
              .distribute_local_to_global(normal_cell_rhs[d], local_dof_indices, rhs.block(d));

        } // end of cell loop
    matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  NormalVectorOperator<dim, number>::vmult(BlockVectorType &dst, const BlockVectorType &src) const
  {
    Assert(!normal_vector_data.narrow_band.enable || solution_level_set,
           ExcMessage(
             "Level set solution vector must not be nullptr for a narrow band computation."));

    scratch_data.get_matrix_free().template cell_loop<BlockVectorType, BlockVectorType>(
      [&](const auto &, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> normal(scratch_data.get_matrix_free(),
                                                normal_dof_idx,
                                                normal_quad_idx);
        FECellIntegrator<dim, 1, number> level_set(scratch_data.get_matrix_free(),
                                                   ls_dof_idx,
                                                   normal_quad_idx);
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            normal.reinit(cell);

            for (unsigned int b = 0; b < dim; ++b)
              {
                normal.read_dof_values(src.block(b));

                tangent_local_cell_operation(normal, level_set, b == 0);

                normal.distribute_local_to_global(dst.block(b));
              }
          }
      },
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  NormalVectorOperator<dim, number>::create_rhs(BlockVectorType &dst, const VectorType &src) const
  {
    FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                     normal_dof_idx,
                                                     normal_quad_idx);
    FECellIntegrator<dim, 1, number>   level_set(scratch_data.get_matrix_free(),
                                               ls_dof_idx,
                                               normal_quad_idx);

    scratch_data.get_matrix_free().template cell_loop<BlockVectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto macro_cells) {
        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            normal_vector.reinit(cell);

            level_set.reinit(cell);
            level_set.read_dof_values_plain(src);
            level_set.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < normal_vector.n_q_points; ++q_index)
              {
                const VectorizedArray<number> narrow_band_mask =
                  (normal_vector_data.narrow_band.enable) ?
                    VectorTools::compute_mask_narrow_band<dim>(
                      level_set.get_value(q_index),
                      normal_vector_data.narrow_band.level_set_threshold) :
                    1.0;

                normal_vector.submit_value(narrow_band_mask * level_set.get_gradient(q_index),
                                           q_index);
              }

            normal_vector.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  NormalVectorOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    // note: not thread safe!!!
    const auto                      &matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, 1, number> level_set_vals(matrix_free, ls_dof_idx, normal_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute matrix (only cell contributions)
    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      scratch_data.get_constraint(normal_dof_idx),
      system_matrix,
      [&](auto &normal_vals) {
        const unsigned int current_cell_index = normal_vals.get_current_cell_index();

        tangent_local_cell_operation(normal_vals,
                                     level_set_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      normal_dof_idx,
      normal_quad_idx);

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  NormalVectorOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    BlockVectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, normal_dof_idx);

    bool update_ghosts = true;

    if (solution_level_set)
      {
        update_ghosts = !solution_level_set->has_ghost_elements();
        if (update_ghosts)
          solution_level_set->update_ghost_values();
      }

    // note: not thread safe!!!
    const auto                      &matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, 1, number> level_set_vals(scratch_data.get_matrix_free(),
                                                    ls_dof_idx,
                                                    normal_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute diagonal ...
    for (unsigned int b = 0; b < dim; ++b)
      MatrixFreeTools::
        template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>, VectorType>(
          matrix_free,
          diagonal.block(b),
          [&](auto &normal_vals) {
            const unsigned int current_cell_index = normal_vals.get_current_cell_index();

            tangent_local_cell_operation(normal_vals,
                                         level_set_vals,
                                         old_cell_index != current_cell_index);

            old_cell_index = current_cell_index;
          },
          normal_dof_idx,
          normal_quad_idx);

    // ... and invert it
    const double linfty_norm = std::max(1.0, diagonal.linfty_norm());

    for (unsigned int d = 0; d < dim; ++d)
      for (auto &i : diagonal.block(d))
        i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  NormalVectorOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number> &normal_vector_vals,
    FECellIntegrator<dim, 1, number> &level_set_vals,
    const bool                        do_reinit_cells) const
  {
    normal_vector_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (do_reinit_cells && normal_vector_data.narrow_band.enable)
      {
        level_set_vals.reinit(normal_vector_vals.get_current_cell_index());
        level_set_vals.read_dof_values_plain(*solution_level_set);
        level_set_vals.evaluate(EvaluationFlags::values);
      }

    for (unsigned int q_index = 0; q_index < normal_vector_vals.n_q_points; ++q_index)
      {
        const auto grad_val = damping[normal_vector_vals.get_current_cell_index()] *
                              normal_vector_vals.get_gradient(q_index);

        const VectorizedArray<number> narrow_band_mask =
          (normal_vector_data.narrow_band.enable) ?
            VectorTools::compute_mask_narrow_band<dim>(
              level_set_vals.get_value(q_index),
              normal_vector_data.narrow_band.level_set_threshold) :
            1.0;

        normal_vector_vals.submit_value(narrow_band_mask * normal_vector_vals.get_value(q_index),
                                        q_index);
        normal_vector_vals.submit_gradient(narrow_band_mask * grad_val, q_index);
      }

    normal_vector_vals.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  NormalVectorOperator<dim, number>::get_unit_normals_at_quadrature(
    const FEValues<dim>         &fe_values,
    const BlockVectorType       &normal_vector_field_in,
    std::vector<Tensor<1, dim>> &unit_normal_at_quadrature,
    const double                 zero)
  {
    for (unsigned int d = 0; d < dim; ++d)
      {
        std::vector<double> temp(unit_normal_at_quadrature.size());
        fe_values.get_function_values(normal_vector_field_in.block(d),
                                      temp); // compute normals from level set solution at tau=0
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          unit_normal_at_quadrature[q_index][d] = temp[q_index];
      }
    for (auto &n : unit_normal_at_quadrature)
      {
        const double n_norm = n.norm();
        if (n_norm > zero)
          n /= n_norm; //@todo: add exception if norm is zero
        else
          n = 0.0;
      }
  }

  template <int dim, typename number>
  void
  NormalVectorOperator<dim, number>::reinit()
  {
    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        damping.resize(scratch_data.get_matrix_free().n_cell_batches());

        for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
          {
            damping[cell] = compute_cell_size_dependent_filter_parameter_mf<dim>(
              scratch_data, normal_dof_idx, cell, normal_vector_data.filter_parameter);
          }
      }
  }

  template class NormalVectorOperator<1, double>;
  template class NormalVectorOperator<2, double>;
  template class NormalVectorOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
