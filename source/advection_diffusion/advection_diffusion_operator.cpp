#include <meltpooldg/advection_diffusion/advection_diffusion_operator.hpp>
//

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/interface/exceptions.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  AdvectionDiffusionOperator<dim, number>::AdvectionDiffusionOperator(
    const ScratchData<dim>               &scratch_data_in,
    const VectorType                     &advection_velocity_in,
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
          "Advection diffusion operator: Requested time integration scheme not supported."));
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::assemble_matrixbased(
    const VectorType                             &advected_field_old,
    AdvectionDiffusionOperator::SparseMatrixType &matrix,
    VectorType                                   &rhs) const
  {
    AssertThrowZeroTimeIncrement(this->time_increment);

    const FEValuesExtractors::Vector velocities(0);

    const bool vel_update_ghosts = !advection_velocity.has_ghost_elements();
    if (vel_update_ghosts)
      advection_velocity.update_ghost_values();

    const bool update_ghosts = !advected_field_old.has_ghost_elements();
    if (update_ghosts)
      advected_field_old.update_ghost_values();

    FEValues<dim> advec_diff_values(scratch_data.get_mapping(),
                                    scratch_data.get_dof_handler(this->dof_idx).get_fe(),
                                    scratch_data.get_quadrature(advec_diff_quad_idx),
                                    update_values | update_gradients | update_quadrature_points |
                                      update_JxW_values);

    FEValues<dim> vel_values(scratch_data.get_mapping(),
                             scratch_data.get_dof_handler(velocity_dof_idx).get_fe(),
                             scratch_data.get_quadrature(advec_diff_quad_idx),
                             update_values);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(this->dof_idx);

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

    if (vel_update_ghosts)
      advection_velocity.zero_out_ghost_values();

    if (update_ghosts)
      advected_field_old.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::reinit()
  {
    if (data.conv_stab.type == ConvectionStabilizationType::SUPG)
      stab_param.resize_fast(scratch_data.get_matrix_free().n_cell_batches());
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::prepare()
  {
    do_update_stab_param = true;
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    if (do_pre_post)
      do_pre_vmult(dst, src);

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

    if (do_pre_post)
      do_post_vmult(dst, src);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::create_rhs(
    VectorType       &dst,
    const VectorType &src /* old advected field*/) const
  {
    /*
     * This function creates the rhs of the advection-diffusion problem. When inhomogeneous
     * dirichlet BC are prescribed, the rhs vector is modified including BC terms. Thus the src
     * vector will NOT be zeroed during the cell_loop.
     */
    AssertThrowZeroTimeIncrement(this->time_increment);

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

            if (data.conv_stab.type == ConvectionStabilizationType::SUPG)
              advected_field_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients |
                                           EvaluationFlags::hessians);
            else
              advected_field_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            velocity_vals.reinit(cell);
            velocity_vals.read_dof_values_plain(advection_velocity);
            velocity_vals.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < advected_field_vals.n_q_points; ++q_index)
              {
                scalar       phi      = advected_field_vals.get_value(q_index);
                const vector grad_phi = advected_field_vals.get_gradient(q_index);
                const vector vel = VectorTools::to_vector<dim>(velocity_vals.get_value(q_index));

                const scalar velocity_grad_phi = scalar_product(vel, grad_phi);

                scalar res_val = this->time_increment_inv * phi - (1. - theta) * velocity_grad_phi;

                vector res_grad = -(1. - theta) * data.diffusivity * grad_phi;

                advected_field_vals.submit_value(res_val, q_index);

                if (data.conv_stab.type == ConvectionStabilizationType::SUPG)
                  {
                    res_val +=
                      (1. - theta) * data.diffusivity * advected_field_vals.get_laplacian(q_index);
                    res_grad += stab_param[cell] * vel * res_val;
                  }

                advected_field_vals.submit_gradient(res_grad, q_index);
              }

            advected_field_vals.integrate_scatter(EvaluationFlags::values |
                                                    EvaluationFlags::gradients,
                                                  dst);
          }
      },
      dst,
      src,
      false); // rhs should not be zeroed out in order to consider inhomogeneous dirichlet BC

    do_update_stab_param = false;
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    // note: not thread safe!!!
    const auto                        &matrix_free = scratch_data.get_matrix_free();
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

    if (do_pre_post)
      post_system_matrix_compute(system_matrix);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);

    // note: not thread safe!!!
    const auto                        &matrix_free = scratch_data.get_matrix_free();
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

    if (do_pre_post)
      {
        for (const auto &i : inflow_outflow_bc_local_indices)
          diagonal.local_element(i) = 1.0;
      }
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number>   &advected_field_vals,
    FECellIntegrator<dim, dim, number> &velocity_vals,
    const bool                          do_reinit_cells) const
  {
    if (data.conv_stab.type == ConvectionStabilizationType::SUPG)
      advected_field_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients |
                                   EvaluationFlags::hessians);
    else
      advected_field_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (do_reinit_cells)
      {
        velocity_vals.reinit(advected_field_vals.get_current_cell_index());
        velocity_vals.read_dof_values_plain(advection_velocity);
        velocity_vals.evaluate(EvaluationFlags::values);
      }

    if (do_update_stab_param)
      compute_stabilization_parameter(velocity_vals);

    for (unsigned int q_index = 0; q_index < advected_field_vals.n_q_points; ++q_index)
      {
        const scalar phi      = advected_field_vals.get_value(q_index);
        const vector grad_phi = advected_field_vals.get_gradient(q_index);
        const vector vel      = VectorTools::to_vector<dim>(velocity_vals.get_value(q_index));
        const scalar velocity_grad_phi = scalar_product(vel, grad_phi);

        scalar res_val  = this->time_increment_inv * phi + theta * velocity_grad_phi;
        vector res_grad = theta * data.diffusivity * grad_phi;

        advected_field_vals.submit_value(res_val, q_index);
        if (data.conv_stab.type == ConvectionStabilizationType::SUPG)
          {
            res_val -= theta * data.diffusivity * advected_field_vals.get_laplacian(q_index);
            res_grad += stab_param[advected_field_vals.get_current_cell_index()] * vel * res_val;
          }

        advected_field_vals.submit_gradient(res_grad, q_index);
      }
    advected_field_vals.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::compute_stabilization_parameter(
    const FECellIntegrator<dim, dim, number> &velocity_vals) const
  {
    if (data.conv_stab.type == ConvectionStabilizationType::SUPG)
      {
        const unsigned int cell = velocity_vals.get_current_cell_index();

        if (data.conv_stab.coefficient < 0.0)
          {
            stab_param[cell] = 0.0;

            for (unsigned int q_index = 0; q_index < velocity_vals.n_q_points; ++q_index)
              stab_param[cell] =
                std::max(stab_param[cell],
                         VectorTools::to_vector<dim>(velocity_vals.get_value(q_index)).norm());

            stab_param[cell] = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
              stab_param[cell], 1e-6, 1. / stab_param[cell], 0.0);
          }
        else
          {
            stab_param[cell] = data.conv_stab.coefficient;
          }
        stab_param[cell] *= scratch_data.get_cell_sizes()[velocity_vals.get_current_cell_index()];
      }
  }


  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::set_inflow_outflow_bc(
    std::vector<unsigned int> inflow_outflow_bc_local_indices_)
  {
    inflow_outflow_bc_local_indices = inflow_outflow_bc_local_indices_;
  }



  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::enable_pre_post()
  {
    do_pre_post = true;
  }



  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::disable_pre_post()
  {
    do_pre_post = false;
  }



  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::do_pre_vmult([[maybe_unused]] VectorType &dst,
                                                        const VectorType            &src_in) const
  {
    VectorType &src = const_cast<VectorType &>(src_in);

    inflow_outflow_constraints_values_temp.resize(inflow_outflow_bc_local_indices.size());

    for (unsigned int i = 0; i < inflow_outflow_bc_local_indices.size(); ++i)
      {
        const unsigned int index                  = inflow_outflow_bc_local_indices[i];
        inflow_outflow_constraints_values_temp[i] = src.local_element(index);
        src.local_element(index)                  = 0.0;
      }
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::do_post_vmult(
    VectorType                        &dst,
    [[maybe_unused]] const VectorType &src_in) const
  {
    for (unsigned int i = 0; i < inflow_outflow_bc_local_indices.size(); ++i)
      {
        const auto &index        = inflow_outflow_bc_local_indices[i];
        const auto &value        = inflow_outflow_constraints_values_temp[i];
        dst.local_element(index) = value;
      }
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperator<dim, number>::post_system_matrix_compute(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    const auto &partitioner = scratch_data.get_matrix_free().get_vector_partitioner(this->dof_idx);

    for (const auto &i : inflow_outflow_bc_local_indices)
      {
        const auto global_index = partitioner->local_to_global(i);
        system_matrix.clear_row(global_index, 1.0);
      }
  }


  template class AdvectionDiffusionOperator<1, double>;
  template class AdvectionDiffusionOperator<2, double>;
  template class AdvectionDiffusionOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
