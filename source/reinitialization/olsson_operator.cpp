#include <meltpooldg/reinitialization/olsson_operator.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Reinitialization
{
  template <int dim, typename number>
  OlssonOperator<dim, number>::OlssonOperator(const ScratchData<dim> &scratch_data_in,
                                              BlockVectorType &       n_in,
                                              const double &          constant_epsilon,
                                              const double &          eps_scale_factor,
                                              const unsigned int      dof_idx_in,
                                              const unsigned int      quad_idx_in,
                                              const unsigned int      ls_dof_idx_in,
                                              const unsigned int      normal_dof_idx_in)
    : scratch_data(scratch_data_in)
    , eps(constant_epsilon)
    , eps_scale_factor(eps_scale_factor)
    , epsilon_used(constant_epsilon > 0 ?
                     constant_epsilon :
                     eps_scale_factor *
                       scratch_data.get_min_cell_size()) // @ todo: check how cell size can be
    , normal_vec(n_in)
    , tolerance_normal_vector(
        UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                              scratch_data.get_mapping()))
    , reinit_quad_idx(quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
  {
    this->reset_dof_index(dof_idx_in);

    Assert(epsilon_used > 0.0, ExcMessage("reinitialization operator: epsilon must be set"));
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::assemble_matrixbased(const VectorType &levelset_old,
                                                    SparseMatrixType &matrix,
                                                    VectorType &      rhs) const
  {
    levelset_old.update_ghost_values();

    FEValues<dim>      fe_values(scratch_data.get_mapping(),
                            scratch_data.get_dof_handler(this->dof_idx).get_fe(),
                            scratch_data.get_quadrature(reinit_quad_idx),
                            update_values | update_gradients | update_quadrature_points |
                              update_JxW_values);
    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    const unsigned int n_q_points = fe_values.get_quadrature().size();

    std::vector<double>         psi_at_q(n_q_points);
    std::vector<Tensor<1, dim>> grad_psi_at_q(n_q_points, Tensor<1, dim>());
    std::vector<Tensor<1, dim>> normal_at_q(n_q_points, Tensor<1, dim>());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    rhs    = 0.0;
    matrix = 0.0;

    this->normal_vec.update_ghost_values();

    for (const auto &cell : scratch_data.get_dof_handler(this->dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.0;
          cell_rhs    = 0.0;
          fe_values.reinit(cell);

          const double epsilon_cell =
            eps > 0.0 ? eps :
                        UtilityFunctions::compute_cell_size_dependent_interface_thickness<dim>(
                          cell, eps_scale_factor);
          AssertThrow(
            epsilon_cell > 0.0,
            ExcMessage(
              "Reinitialization: the value of epsilon for the reinitialization function must be larger than zero!"));

          fe_values.get_function_values(levelset_old,
                                        psi_at_q); // compute values of old solution at tau_n
          fe_values.get_function_gradients(
            levelset_old, grad_psi_at_q); // compute gradients of old solution at tau_n
          NormalVector::NormalVectorOperator<dim>::get_unit_normals_at_quadrature(
            fe_values, this->normal_vec, normal_at_q, tolerance_normal_vector);

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                {
                  const double nTimesGradient_i =
                    normal_at_q[q_index] * fe_values.shape_grad(i, q_index);

                  for (const unsigned int j : fe_values.dof_indices())
                    {
                      const double nTimesGradient_j =
                        normal_at_q[q_index] * fe_values.shape_grad(j, q_index);
                      //clang-format off
                      cell_matrix(i, j) +=
                        (fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index) +
                         this->time_increment * epsilon_cell * nTimesGradient_i *
                           nTimesGradient_j) *
                        fe_values.JxW(q_index);
                      //clang-format on
                    }

                  const double diffRhs =
                    epsilon_cell * normal_at_q[q_index] * grad_psi_at_q[q_index];

                  //clang-format off
                  const auto compressive_flux = [](const double psi) {
                    return 0.5 * (1. - psi * psi);
                  };
                  cell_rhs(i) += (compressive_flux(psi_at_q[q_index]) - diffRhs) *
                                 nTimesGradient_i * this->time_increment * fe_values.JxW(q_index);
                  //clang-format on
                }
            } // end loop over gauss points
          // assembly
          cell->get_dof_indices(local_dof_indices);

          scratch_data.get_constraint(this->dof_idx)
            .distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, matrix, rhs);
        }

    matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);

    this->normal_vec.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    AssertThrow(this->time_increment > 0.0,
                ExcMessage("reinitialization operator: d_tau must be set"));

    this->normal_vec.update_ghost_values();

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number>   delta_psi(scratch_data.get_matrix_free(),
                                                   this->dof_idx,
                                                   reinit_quad_idx);
        FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                         normal_dof_idx,
                                                         reinit_quad_idx);
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            delta_psi.reinit(cell);
            delta_psi.read_dof_values(src);

            tangent_local_cell_operation(delta_psi, normal_vector, true);

            delta_psi.distribute_local_to_global(dst);
          }
      },
      dst,
      src,
      true);

    this->normal_vec.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::create_rhs(VectorType &dst, const VectorType &src) const
  {
    this->normal_vec.update_ghost_values();

    AssertThrow(this->time_increment > 0.0,
                ExcMessage("reinitialization matrix-free operator: d_tau must be set"));

    const auto compressive_flux = [&](const auto &phi) {
      return 0.5 * (make_vectorized_array<number>(1.) - phi * phi);
    };

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number>   rhs(scratch_data.get_matrix_free(),
                                             this->dof_idx,
                                             reinit_quad_idx);
        FECellIntegrator<dim, 1, number>   psi_old(scratch_data.get_matrix_free(),
                                                 ls_dof_idx,
                                                 reinit_quad_idx);
        FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                         normal_dof_idx,
                                                         reinit_quad_idx);
        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            rhs.reinit(cell);

            psi_old.reinit(cell);
            psi_old.read_dof_values_plain(src);
            psi_old.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            normal_vector.reinit(cell);
            normal_vector.read_dof_values_plain(this->normal_vec);
            normal_vector.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < rhs.n_q_points; ++q_index)
              {
                const scalar val = psi_old.get_value(q_index);
                const auto   n_phi =
                  MeltPoolDG::VectorTools::normalize<dim>(normal_vector.get_value(q_index),
                                                          tolerance_normal_vector);

                rhs.submit_gradient(this->time_increment * compressive_flux(val) * n_phi -
                                      this->time_increment * epsilon_used *
                                        scalar_product(psi_old.get_gradient(q_index), n_phi) *
                                        n_phi,
                                    q_index);
              }

            rhs.integrate_scatter(EvaluationFlags::gradients, dst);
          }
      },
      dst,
      src,
      true);

    this->normal_vec.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::set_normal_vector_field(const BlockVectorType &normal_vector)
  {
    normal_vec.reinit(dim);
    for (unsigned int d = 0; d < dim; ++d)
      {
        scratch_data.initialize_dof_vector(this->normal_vec.block(d), this->dof_idx);
        normal_vec.block(d).copy_locally_owned_data_from(normal_vector.block(d));
      }
    normal_vec.update_ghost_values();
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    this->normal_vec.update_ghost_values();

    // note: not thread safe!!!
    const auto &                       matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                     normal_dof_idx,
                                                     reinit_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute matrix (only cell contributions)
    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      scratch_data.get_constraint(this->dof_idx),
      system_matrix,
      [&](auto &delta_psi) {
        const unsigned int current_cell_index = delta_psi.get_current_cell_index();

        tangent_local_cell_operation(delta_psi,
                                     normal_vector,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      this->dof_idx,
      reinit_quad_idx);

    system_matrix.compress(VectorOperation::add);

    this->normal_vec.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);

    this->normal_vec.update_ghost_values();

    // note: not thread safe!!!
    const auto &                       matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                     normal_dof_idx,
                                                     reinit_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute diagonal ...
    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal,
      [&](auto &delta_psi) {
        const unsigned int current_cell_index = delta_psi.get_current_cell_index();

        tangent_local_cell_operation(delta_psi,
                                     normal_vector,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      this->dof_idx,
      reinit_quad_idx);

    // ... and invert it
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-16) ? (1.0 / i) : 1.0;

    this->normal_vec.zero_out_ghost_values();
  }


  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number> &  delta_psi,
    FECellIntegrator<dim, dim, number> &normal_vector,
    const bool                          do_reinit_cells) const
  {
    delta_psi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (do_reinit_cells)
      {
        normal_vector.reinit(delta_psi.get_current_cell_index());
        normal_vector.read_dof_values_plain(this->normal_vec);
        normal_vector.evaluate(EvaluationFlags::values);
      }

    for (unsigned int q_index = 0; q_index < delta_psi.n_q_points; q_index++)
      {
        const auto n_phi = MeltPoolDG::VectorTools::normalize<dim>(normal_vector.get_value(q_index),
                                                                   tolerance_normal_vector);

        delta_psi.submit_value(delta_psi.get_value(q_index), q_index);
        delta_psi.submit_gradient(this->time_increment * epsilon_used *
                                    scalar_product(delta_psi.get_gradient(q_index), n_phi) * n_phi,
                                  q_index);
      }

    delta_psi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }



  template class OlssonOperator<1, double>;
  template class OlssonOperator<2, double>;
  template class OlssonOperator<3, double>;
} // namespace MeltPoolDG::Reinitialization
