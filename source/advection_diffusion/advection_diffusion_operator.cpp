#include <meltpooldg/advection_diffusion/advection_diffusion_operator.hpp>
//

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/interface/exceptions.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::AdvectionDiffusion
{
  template <int dim, typename number>
  AdvectionDiffusionOperator<dim, number>::AdvectionDiffusionOperator(
    const ScratchData<dim> &              scratch_data_in,
    const VectorType &                    advection_velocity_in,
    const AdvectionDiffusionData<number> &data_in,
    const unsigned int                    dof_idx_in,
    const unsigned int                    quad_idx_in,
    const unsigned int                    velocity_dof_idx_in)
    : scratch_data(scratch_data_in)
    , advection_velocity(advection_velocity_in)
    , data(data_in)
    , velocity_dof_idx(velocity_dof_idx_in)
    , advec_diff_quad_idx(quad_idx_in)
  {
    this->reset_dof_index(dof_idx_in);
    /*
     *  convert the user input to the generalized theta parameter
     */
    if (get_generalized_theta.find(data.time_integration_scheme) != get_generalized_theta.end())
      theta = get_generalized_theta[data.time_integration_scheme];
    else
      AssertThrow(
        false,
        ExcMessage(
          "Advection diffusion operator: Requested time integration scheme not supported."))
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::assemble_matrixbased(
    const VectorType &                            advected_field_old,
    AdvectionDiffusionOperator::SparseMatrixType &matrix,
    VectorType &                                  rhs) const
  {
    AssertThrow(this->time_increment > 0.0, ExcZeroTimeIncrement());

    AssertThrow(data.diffusivity >= 0.0,
                ExcMessage("Advection diffusion operator: diffusivity is smaller than zero!"));

    const FEValuesExtractors::Vector velocities(0);

    FEValues<dim> advec_diff_values(scratch_data.get_mapping(),
                                    scratch_data.get_dof_handler(this->dof_idx).get_fe(),
                                    scratch_data.get_quadrature(advec_diff_quad_idx),
                                    update_values | update_gradients | update_quadrature_points |
                                      update_JxW_values);

    FEValues<dim> vel_values(scratch_data.get_mapping(),
                             scratch_data.get_dof_handler(velocity_dof_idx).get_fe(),
                             scratch_data.get_quadrature(advec_diff_quad_idx),
                             update_values);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    const unsigned int n_q_points = advec_diff_values.get_quadrature().size();

    std::vector<double>         phi_at_q(n_q_points);
    std::vector<Tensor<1, dim>> grad_phi_at_q(n_q_points, Tensor<1, dim>());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    rhs    = 0.0;
    matrix = 0.0;

    std::vector<Tensor<1, dim>> a(n_q_points, Tensor<1, dim>());

    typename DoFHandler<dim>::active_cell_iterator vel_cell =
      scratch_data.get_dof_handler(velocity_dof_idx).begin_active();

    for (const auto &cell : scratch_data.get_dof_handler(this->dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell_matrix = 0;
            cell_rhs    = 0;

            advec_diff_values.reinit(cell);
            advec_diff_values.get_function_values(advected_field_old, phi_at_q);
            advec_diff_values.get_function_gradients(advected_field_old, grad_phi_at_q);

            vel_values.reinit(vel_cell);
            std::vector<Tensor<1, dim>> a(n_q_points, Tensor<1, dim>());
            vel_values[velocities].get_function_values(advection_velocity, a);

            for (const unsigned int q_index : advec_diff_values.quadrature_point_indices())
              {
                for (const unsigned int i : advec_diff_values.dof_indices())
                  {
                    for (const unsigned int j : advec_diff_values.dof_indices())
                      {
                        auto velocity_grad_phi_j =
                          a[q_index] * advec_diff_values.shape_grad(j, q_index);
                        // clang-format off
              cell_matrix( i, j ) +=
                ( advec_diff_values.shape_value( i, q_index)
                  *
                  advec_diff_values.shape_value( j, q_index)
                  * 
                  this->time_increment_inv 
                  +
                  theta * ( data.diffusivity *
                                          advec_diff_values.shape_grad( i, q_index) *
                                          advec_diff_values.shape_grad( j, q_index)
                                          +
                                          advec_diff_values.shape_value( i, q_index)  *
                                          velocity_grad_phi_j )
                ) * advec_diff_values.JxW(q_index);
                        // clang-format on
                      }

                    // clang-format off
            cell_rhs( i ) +=
              (  advec_diff_values.shape_value( i, q_index) * phi_at_q[q_index] * this->time_increment_inv 
                 -
                 ( 1. - theta ) *
                 (
                   data.diffusivity * advec_diff_values.shape_grad( i, q_index) * grad_phi_at_q[q_index]
                   +
                   advec_diff_values.shape_value(i, q_index) * a[q_index] * grad_phi_at_q[q_index]
                 )
              ) * advec_diff_values.JxW(q_index) ;
                    // clang-format on
                  }
              } // end gauss

            // assembly
            cell->get_dof_indices(local_dof_indices);

            scratch_data.get_constraint(this->dof_idx)
              .distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, matrix, rhs);
          }
        ++vel_cell;
      }

    matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number>   advected_field_vals(matrix_free,
                                                             this->dof_idx,
                                                             advec_diff_quad_idx);
        FECellIntegrator<dim, dim, number> velocity_vals(matrix_free,
                                                         velocity_dof_idx,
                                                         advec_diff_quad_idx);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            advected_field_vals.reinit(cell);
            advected_field_vals.read_dof_values(src);

            tangent_local_cell_operation(advected_field_vals, velocity_vals, true);

            advected_field_vals.distribute_local_to_global(dst);
          }
      },
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::create_rhs(
    VectorType &      dst,
    const VectorType &src /* old advected field*/) const
  {
    /*
     * This function creates the rhs of the advection-diffusion problem. When inhomogeneous
     * dirichlet BC are prescribed, the rhs vector is modified including BC terms. Thus the src
     * vector will NOT be zeroed during the cell_loop.
     */
    AssertThrow(this->time_increment > 0.0, ExcZeroTimeIncrement());

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number, VectorizedArrayType> advected_field_vals(
          matrix_free, this->dof_idx, advec_diff_quad_idx);
        FECellIntegrator<dim, dim, number> velocity_vals(matrix_free,
                                                         velocity_dof_idx,
                                                         advec_diff_quad_idx);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            advected_field_vals.reinit(cell);
            advected_field_vals.read_dof_values_plain(src);
            advected_field_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            velocity_vals.reinit(cell);
            velocity_vals.read_dof_values_plain(advection_velocity);
            velocity_vals.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < advected_field_vals.n_q_points; ++q_index)
              {
                scalar       phi      = advected_field_vals.get_value(q_index);
                const vector grad_phi = advected_field_vals.get_gradient(q_index);

                const scalar velocity_grad_phi =
                  scalar_product(velocity_vals.get_value(q_index), grad_phi);
                advected_field_vals.submit_value(this->time_increment_inv * phi -
                                                   (1. - theta) * velocity_grad_phi,
                                                 q_index);

                advected_field_vals.submit_gradient(-(1. - theta) * data.diffusivity * grad_phi,
                                                    q_index);
              }

            advected_field_vals.integrate_scatter(EvaluationFlags::values |
                                                    EvaluationFlags::gradients,
                                                  dst);
          }
      },
      dst,
      src,
      false); // rhs should not be zeroed out in order to consider inhomogeneous dirichlet BC
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    // note: not thread safe!!!
    const auto &                       matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free,
                                                     velocity_dof_idx,
                                                     advec_diff_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute matrix (only cell contributions)
    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      scratch_data.get_constraint(this->dof_idx),
      system_matrix,
      [&](auto &advected_field_vals) {
        const unsigned int current_cell_index = advected_field_vals.get_current_cell_index();

        tangent_local_cell_operation(advected_field_vals,
                                     velocity_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      this->dof_idx,
      advec_diff_quad_idx);

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);

    // note: not thread safe!!!
    const auto &                       matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free,
                                                     velocity_dof_idx,
                                                     advec_diff_quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute diagonal ...
    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal,
      [&](auto &advected_field_vals) {
        const unsigned int current_cell_index = advected_field_vals.get_current_cell_index();

        tangent_local_cell_operation(advected_field_vals,
                                     velocity_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      this->dof_idx,
      advec_diff_quad_idx);

    // ... and invert it
    const double linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }


  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number> &  advected_field_vals,
    FECellIntegrator<dim, dim, number> &velocity_vals,
    const bool                          do_reinit_cells) const
  {
    advected_field_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (do_reinit_cells)
      {
        velocity_vals.reinit(advected_field_vals.get_current_cell_index());
        velocity_vals.read_dof_values_plain(advection_velocity);
        velocity_vals.evaluate(EvaluationFlags::values);
      }

    for (unsigned int q_index = 0; q_index < advected_field_vals.n_q_points; ++q_index)
      {
        const scalar phi      = advected_field_vals.get_value(q_index);
        const vector grad_phi = advected_field_vals.get_gradient(q_index);

        const scalar velocity_grad_phi = scalar_product(velocity_vals.get_value(q_index), grad_phi);

        advected_field_vals.submit_value(this->time_increment_inv * phi + theta * velocity_grad_phi,
                                         q_index);
        advected_field_vals.submit_gradient(theta * data.diffusivity * grad_phi, q_index);
      }
    advected_field_vals.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template class AdvectionDiffusionOperator<1, double>;
  template class AdvectionDiffusionOperator<2, double>;
  template class AdvectionDiffusionOperator<3, double>;
} // namespace MeltPoolDG::AdvectionDiffusion
