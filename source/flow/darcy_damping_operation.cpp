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
                // K = -C * fs² / ( (1-fs)³ + b )
                // K := permeability,  C := morphology
                // b := avoid div zero constant, fs := solid fraction
                const auto non_solid = 1.0 - solid.get_value(q_index);
                const auto permeability =
                  -mushy_zone_morphology * solid.get_value(q_index) * solid.get_value(q_index) /
                  (non_solid * non_solid * non_solid + avoid_div_zero_constant);
                darcy_damping_force.submit_value(permeability * velocity.get_value(q_index),
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

  template class DarcyDampingOperation<1>;
  template class DarcyDampingOperation<2>;
  template class DarcyDampingOperation<3>;
} // namespace MeltPoolDG::Flow
