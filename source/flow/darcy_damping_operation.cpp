#include <meltpooldg/flow/darcy_damping_operation.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim>
  DarcyDampingOperation<dim>::DarcyDampingOperation(
    const DarcyDampingData<double> &data_in,
    const ScratchData<dim> &        scratch_data,
    const unsigned int              flow_vel_hanging_nodes_dof_idx,
    const unsigned int              flow_quad_idx,
    const unsigned int              solid_dof_idx)
    : mushy_zone_morphology(data_in.mushy_zone_morphology)
    , avoid_div_zero_constant(data_in.avoid_div_zero_constant)
    , scratch_data(scratch_data)
    , flow_vel_hanging_nodes_dof_idx(flow_vel_hanging_nodes_dof_idx)
    , flow_quad_idx(flow_quad_idx)
    , solid_dof_idx(solid_dof_idx)
  {
    AssertThrow(mushy_zone_morphology == 0.0 || avoid_div_zero_constant > 0.0,
                ExcMessage(
                  "When using the Darcy damping force, the parameter \"mp solid "
                  "darcy damping avoid div zero constant\" must be greater than zero! Abort.."));
  }

  template <int dim>
  void
  DarcyDampingOperation<dim>::compute_darcy_damping(VectorType &      force_rhs,
                                                    const VectorType &velocity_vec,
                                                    const VectorType &solid_fraction_vec,
                                                    const bool        zero_out)
  {
    velocity_vec.update_ghost_values();

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto &      force_rhs,
          const auto &solid_fraction_vec,
          auto        macro_cells) {
        FECellIntegrator<dim, 1, double> solid(matrix_free, solid_dof_idx, flow_quad_idx);

        FECellIntegrator<dim, dim, double> velocity(matrix_free,
                                                    flow_vel_hanging_nodes_dof_idx,
                                                    flow_quad_idx);

        FECellIntegrator<dim, dim, double> darcy_damping_force(matrix_free,
                                                               flow_vel_hanging_nodes_dof_idx,
                                                               flow_quad_idx);

        damping_at_q.resize(scratch_data.get_matrix_free().n_cell_batches(),
                            std::vector<VectorizedArray<double>>(solid.n_q_points));

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            solid.reinit(cell);
            solid.read_dof_values_plain(solid_fraction_vec);
            solid.evaluate(EvaluationFlags::values);

            velocity.reinit(cell);
            velocity.read_dof_values_plain(velocity_vec);
            velocity.evaluate(EvaluationFlags::values);

            darcy_damping_force.reinit(cell);

            for (unsigned int q_index = 0; q_index < darcy_damping_force.n_q_points; ++q_index)
              {
                damping_at_q[cell][q_index] =
                  get_darcy_damping_coefficient(solid.get_value(q_index));

                darcy_damping_force.submit_value(damping_at_q[cell][q_index] *
                                                   velocity.get_value(q_index),
                                                 q_index);
              }
            darcy_damping_force.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      solid_fraction_vec,
      zero_out);

    velocity_vec.zero_out_ghost_values();
  }

  template <int dim>
  void
  DarcyDampingOperation<dim>::compute_darcy_damping(VectorType &      force_rhs,
                                                    const VectorType &velocity_vec,
                                                    const bool        zero_out)
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &force_rhs, const auto &velocity_vec, auto macro_cells) {
        FECellIntegrator<dim, dim, double> velocity(matrix_free,
                                                    flow_vel_hanging_nodes_dof_idx,
                                                    flow_quad_idx);

        FECellIntegrator<dim, dim, double> darcy_damping_force(matrix_free,
                                                               flow_vel_hanging_nodes_dof_idx,
                                                               flow_quad_idx);

        // check if damping_at_q has its correct size
        AssertDimension(damping_at_q.size(), scratch_data.get_matrix_free().n_cell_batches());
        AssertDimension(damping_at_q[0].size(), velocity.n_q_points);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            velocity.reinit(cell);
            velocity.read_dof_values_plain(velocity_vec);
            velocity.evaluate(EvaluationFlags::values);

            darcy_damping_force.reinit(cell);

            for (unsigned int q_index = 0; q_index < darcy_damping_force.n_q_points; ++q_index)
              {
                darcy_damping_force.submit_value(damping_at_q[cell][q_index] *
                                                   velocity.get_value(q_index),
                                                 q_index);
              }
            darcy_damping_force.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      velocity_vec,
      zero_out);
  }

  template <int dim>
  VectorizedArray<double>
  DarcyDampingOperation<dim>::get_darcy_damping_coefficient(
    const VectorizedArray<double> &solid_fraction) const
  {
    // K = -C * fs² / ( (1-fs)³ + b )
    // K := permeability,  C := morphology
    // b := avoid div zero constant, fs := solid fraction
    const auto non_solid = 1.0 - solid_fraction;
    return -mushy_zone_morphology * solid_fraction * solid_fraction /
           (non_solid * non_solid * non_solid + avoid_div_zero_constant);
  }

  template <int dim>
  void
  DarcyDampingOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    /**
     * write conductivity vector to dof vector
     */
    scratch_data.initialize_dof_vector(damping, solid_dof_idx);

    if (!damping_at_q.empty() && scratch_data.is_hex_mesh())
      UtilityFunctions::fill_dof_vector_from_cell_operation<dim, 1>(
        damping,
        scratch_data.get_matrix_free(),
        solid_dof_idx,
        flow_quad_idx,
        [&](const unsigned int cell, const unsigned int quad) -> const VectorizedArray<double> & {
          return damping_at_q[cell][quad];
        });

    data_out.add_data_vector(scratch_data.get_dof_handler(solid_dof_idx), damping, "Darcy_damping");
  }

  template <int dim>
  VectorizedArray<double> &
  DarcyDampingOperation<dim>::get_damping(const unsigned int cell, const unsigned int q)
  {
    return damping_at_q[cell][q];
  }

  template <int dim>
  const VectorizedArray<double> &
  DarcyDampingOperation<dim>::get_damping(const unsigned int cell, const unsigned int q) const
  {
    return damping_at_q[cell][q];
  }

  template <int dim>
  std::vector<std::vector<VectorizedArray<double>>> &
  DarcyDampingOperation<dim>::get_damping_at_q()
  {
    return damping_at_q;
  }

  template class DarcyDampingOperation<1>;
  template class DarcyDampingOperation<2>;
  template class DarcyDampingOperation<3>;
} // namespace MeltPoolDG::Flow
