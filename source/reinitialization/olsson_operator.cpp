#include <meltpooldg/reinitialization/olsson_operator.hpp>

namespace MeltPoolDG::Reinitialization
{
  template <int dim, typename number>
  OlssonOperator<dim, number>::OlssonOperator(const ScratchData<dim> &scratch_data_in,
                                              BlockVectorType &       n_in,
                                              const double &          constant_epsilon,
                                              const double &          eps_scale_factor,
                                              const unsigned int      dof_idx_in,
                                              const unsigned int      quad_idx_in)
    : scratch_data(scratch_data_in)
    , eps(constant_epsilon)
    , eps_scale_factor(eps_scale_factor)
    , normal_vec(n_in)
    , tolerance_normal_vector(
        std::min(1e-2,
                 std::max(std::pow(10,
                                   UtilityFunctions::get_exponent_power_ten(std::pow(
                                     GridTools::volume<dim>(scratch_data.get_triangulation(),
                                                            scratch_data.get_mapping()),
                                     1. / dim))) *
                            1e-3,
                          1e-12)))
  {
    this->reset_indices(dof_idx_in, quad_idx_in);
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
                            scratch_data.get_quadrature(this->quad_idx),
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
            eps > 0.0 ? eps : cell->diameter() / (std::sqrt(dim)) * eps_scale_factor;
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
                         this->d_tau * epsilon_cell * nTimesGradient_i * nTimesGradient_j) *
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
                                 nTimesGradient_i * this->d_tau * fe_values.JxW(q_index);
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

    this->normal_vec.zero_out_ghosts();
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    AssertThrow(this->d_tau > 0.0, ExcMessage("reinitialization operator: d_tau must be set"));

    const double eps_ =
      eps > 0 ?
        eps :
        eps_scale_factor *
          scratch_data.get_min_cell_size(
            this->dof_idx); // @ todo: check how cell size can be extracted from matrix free class

    AssertThrow(eps_ > 0.0, ExcMessage("reinitialization operator: epsilon must be set"));

    this->normal_vec.update_ghost_values();

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number>   levelset(scratch_data.get_matrix_free(),
                                                  this->dof_idx,
                                                  this->quad_idx);
        FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                         this->dof_idx,
                                                         this->quad_idx);
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            levelset.reinit(cell);
            levelset.gather_evaluate(src, true, true);

            normal_vector.reinit(cell);
            normal_vector.read_dof_values_plain(this->normal_vec);
            normal_vector.evaluate(true, false);

            for (unsigned int q_index = 0; q_index < levelset.n_q_points; q_index++)
              {
                const scalar phi = levelset.get_value(q_index);

                const vector grad_phi = levelset.get_gradient(q_index);

                const auto n_phi =
                  MeltPoolDG::VectorTools::normalize<dim>(normal_vector.get_value(q_index),
                                                          tolerance_normal_vector);

                levelset.submit_value(phi, q_index);
                levelset.submit_gradient(this->d_tau * eps_ * scalar_product(grad_phi, n_phi) *
                                           n_phi,
                                         q_index);
              }

            levelset.integrate_scatter(true, true, dst);
          }
      },
      dst,
      src,
      true);

    this->normal_vec.zero_out_ghosts();
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::create_rhs(VectorType &dst, const VectorType &src) const
  {
    this->normal_vec.update_ghost_values();

    AssertThrow(this->d_tau > 0.0,
                ExcMessage("reinitialization matrix-free operator: d_tau must be set"));
    const double eps_ =
      eps > 0 ?
        eps :
        eps_scale_factor *
          scratch_data.get_min_cell_size(
            this->dof_idx); // @ todo: check how cell size can be extracted from matrix free class

    AssertThrow(eps_ > 0.0,
                ExcMessage("reinitialization matrix-free operator: epsilon must be set"));

    const auto compressive_flux = [&](const auto &phi) {
      return 0.5 * (make_vectorized_array<number>(1.) - phi * phi);
    };

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number>   psi(scratch_data.get_matrix_free(),
                                             this->dof_idx,
                                             this->quad_idx);
        FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                         this->dof_idx,
                                                         this->quad_idx);
        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            psi.reinit(cell);
            psi.read_dof_values_plain(src);
            psi.evaluate(true, true);

            normal_vector.reinit(cell);
            normal_vector.read_dof_values_plain(this->normal_vec);
            normal_vector.evaluate(true, false);

            for (unsigned int q_index = 0; q_index < psi.n_q_points; ++q_index)
              {
                const scalar val = psi.get_value(q_index);
                const auto   n_phi =
                  MeltPoolDG::VectorTools::normalize<dim>(normal_vector.get_value(q_index),
                                                          tolerance_normal_vector);

                psi.submit_gradient(this->d_tau * compressive_flux(val) * n_phi -
                                      this->d_tau * eps_ *
                                        scalar_product(psi.get_gradient(q_index), n_phi) * n_phi,
                                    q_index);
              }

            psi.integrate_scatter(false, true, dst);
          }
      },
      dst,
      src,
      true);

    this->normal_vec.zero_out_ghosts();
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

  template class OlssonOperator<1, double>;
  template class OlssonOperator<2, double>;
  template class OlssonOperator<3, double>;
} // namespace MeltPoolDG::Reinitialization
