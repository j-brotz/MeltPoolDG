#include <deal.II/base/exceptions.h>

#include <meltpooldg/flow/darcy_damping_operation.hpp>
#include <meltpooldg/material/material.templates.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>

namespace MeltPoolDG::Flow
{
  template <int dim>
  DarcyDampingOperation<dim>::DarcyDampingOperation(
    const DarcyDampingData<double> &data_in,
    const ScratchData<dim>         &scratch_data,
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
  DarcyDampingOperation<dim>::compute_darcy_damping(VectorType       &force_rhs,
                                                    const VectorType &velocity_vec,
                                                    const VectorType &solid_fraction_vec,
                                                    const bool        zero_out)
  {
    const bool update_ghosts = !velocity_vec.has_ghost_elements();

    if (update_ghosts)
      velocity_vec.update_ghost_values();

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto       &force_rhs,
          const auto &solid_fraction_vec,
          auto        macro_cells) {
        FECellIntegrator<dim, 1, double> solid(matrix_free, solid_dof_idx, flow_quad_idx);

        FECellIntegrator<dim, dim, double> velocity(matrix_free,
                                                    flow_vel_hanging_nodes_dof_idx,
                                                    flow_quad_idx);

        FECellIntegrator<dim, dim, double> darcy_damping_force(matrix_free,
                                                               flow_vel_hanging_nodes_dof_idx,
                                                               flow_quad_idx);

        damping_at_q.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                                 darcy_damping_force.n_q_points);

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
                damping_at_q[cell * darcy_damping_force.n_q_points + q_index] =
                  compute_darcy_damping_coefficient(solid.get_value(q_index));

                darcy_damping_force.submit_value(
                  damping_at_q[cell * darcy_damping_force.n_q_points + q_index] *
                    velocity.get_value(q_index),
                  q_index);
              }
            darcy_damping_force.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      solid_fraction_vec,
      zero_out);

    if (update_ghosts)
      velocity_vec.zero_out_ghost_values();
  }

  template <int dim>
  void
  DarcyDampingOperation<dim>::compute_darcy_damping(VectorType       &force_rhs,
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
        AssertDimension(damping_at_q.size(),
                        scratch_data.get_matrix_free().n_cell_batches() *
                          darcy_damping_force.n_q_points);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            velocity.reinit(cell);
            velocity.read_dof_values_plain(velocity_vec);
            velocity.evaluate(EvaluationFlags::values);

            darcy_damping_force.reinit(cell);

            for (unsigned int q_index = 0; q_index < darcy_damping_force.n_q_points; ++q_index)
              {
                darcy_damping_force.submit_value(get_damping(cell, q_index) *
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
  void
  DarcyDampingOperation<dim>::set_darcy_damping_at_q(const Material<double> &material,
                                                     const VectorType       &ls_as_heaviside,
                                                     const VectorType       &temperature,
                                                     const unsigned int ls_hanging_nodes_dof_idx,
                                                     const unsigned int temp_dof_idx)
  {
    // check if damping_at_q has its correct size
    AssertDimension(damping_at_q.size(),
                    scratch_data.get_matrix_free().n_cell_batches() *
                      scratch_data.get_n_q_points(flow_quad_idx));

    double dummy;
    scratch_data.get_matrix_free().template cell_loop<double, VectorType>(
      [&](const auto &matrix_free, auto &, const auto &ls_as_heaviside, auto macro_cells) {
        FECellIntegrator<dim, 1, double> ls_values(matrix_free,
                                                   ls_hanging_nodes_dof_idx,
                                                   flow_quad_idx);

        std::unique_ptr<FECellIntegrator<dim, 1, double>> temp_values;

        if (material.has_dependency(Material<double>::FieldType::temperature))
          temp_values = std::make_unique<FECellIntegrator<dim, 1, double>>(matrix_free,
                                                                           temp_dof_idx,
                                                                           flow_quad_idx);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            ls_values.reinit(cell);
            ls_values.read_dof_values_plain(ls_as_heaviside);
            ls_values.evaluate(EvaluationFlags::values);

            if (temp_values)
              {
                temp_values->reinit(cell);
                temp_values->read_dof_values_plain(temperature);
                temp_values->evaluate(EvaluationFlags::values);
              }

            for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
              {
                const auto material_values =
                  material.template compute_parameters<VectorizedArray<double>>(
                    ls_values, *temp_values, MaterialUpdateFlags::phase_fractions, q);

                get_damping(cell, q) =
                  compute_darcy_damping_coefficient(material_values.solid_fraction);
              }
          }
      },
      dummy,
      ls_as_heaviside);
  }



  template <int dim>
  VectorizedArray<double>
  DarcyDampingOperation<dim>::compute_darcy_damping_coefficient(
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
    if (data_out.is_requested("Darcy_damping"))
      {
        /**
         * write conductivity vector to dof vector
         */
        scratch_data.initialize_dof_vector(damping, solid_dof_idx);

        if (!damping_at_q.empty() && scratch_data.is_hex_mesh())
          {
            MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
              damping,
              scratch_data.get_matrix_free(),
              solid_dof_idx,
              flow_quad_idx,
              [&](const unsigned int cell,
                  const unsigned int quad) -> const VectorizedArray<double> & {
                return damping_at_q[cell * scratch_data.get_n_q_points(flow_quad_idx) + quad];
              });

            scratch_data.get_constraint(solid_dof_idx).distribute(damping);
          }

        data_out.add_data_vector(scratch_data.get_dof_handler(solid_dof_idx),
                                 damping,
                                 "Darcy_damping");
      }
  }
  template <int dim>
  void
  DarcyDampingOperation<dim>::reinit()
  {
    damping_at_q.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                             scratch_data.get_n_q_points(flow_quad_idx));
  }

  template <int dim>
  VectorizedArray<double> &
  DarcyDampingOperation<dim>::get_damping(const unsigned int cell, const unsigned int q)
  {
    return damping_at_q[cell * scratch_data.get_n_q_points(flow_quad_idx) + q];
  }

  template <int dim>
  const VectorizedArray<double> &
  DarcyDampingOperation<dim>::get_damping(const unsigned int cell, const unsigned int q) const
  {
    return damping_at_q[cell * scratch_data.get_n_q_points(flow_quad_idx) + q];
  }

  template <int dim>
  AlignedVector<VectorizedArray<double>> &
  DarcyDampingOperation<dim>::get_damping_at_q()
  {
    return damping_at_q;
  }

  template class DarcyDampingOperation<1>;
  template class DarcyDampingOperation<2>;
  template class DarcyDampingOperation<3>;
} // namespace MeltPoolDG::Flow
