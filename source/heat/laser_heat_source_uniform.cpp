#include <meltpooldg/heat/laser_heat_source_uniform.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserHeatSourceUniform<dim>::LaserHeatSourceUniform(
    const DeltaApproximationPhaseWeightedData<double> &delta_approximation_phase_weighted_data)
    : LaserHeatSourceBase<dim>(delta_approximation_phase_weighted_data)
  {}

  template <int dim>
  double
  LaserHeatSourceUniform<dim>::local_compute_volumetric_heat_source(
    const Point<dim> & /*position*/,
    const Point<dim> & /*laser_position*/,
    const double /*power_density*/) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  template <int dim>
  double
  LaserHeatSourceUniform<dim>::local_compute_interfacial_heat_source(
    const Point<dim> & /*position*/,
    const Point<dim> & /*laser_position*/,
    const double                  power_density,
    const Tensor<1, dim, double> &normal_vector,
    const double                  delta_value,
    const double /*heaviside*/) const
  {
    // assume laser direction coincides with the negative dim-1 direction
    const double projection_factor = std::invoke([&]() {
      const double fac = normal_vector * -Point<dim>::unit_vector(dim - 1);
      if (fac < 0.0)
        return 0.0;
      return fac;
    });

    return projection_factor * delta_value * power_density;
  }

  template <int dim>
  void
  LaserHeatSourceUniform<dim>::compute_interfacial_heat_source_sharp(
    VectorType &            heat_rhs,
    const ScratchData<dim> &scratch_data,
    const unsigned int      temp_dof_idx,
    const double            laser_power_density,
    const Point<dim> &      laser_position,
    const VectorType &      level_set_heaviside,
    const unsigned int      ls_dof_idx,
    const bool              zero_out,
    const BlockVectorType * normal_vector,
    const unsigned int      normal_dof_idx) const
  {
    (void)laser_position;
    if constexpr (dim > 1) // @todo: otherwise a compile error occurs
      {
        if (zero_out)
          scratch_data.initialize_dof_vector(heat_rhs, temp_dof_idx);

        level_set_heaviside.update_ghost_values();
        if (normal_vector)
          normal_vector->update_ghost_values();

        FEPointEvaluation<1, dim> ls(scratch_data.get_mapping(),
                                     scratch_data.get_fe(ls_dof_idx),
                                     normal_vector ? update_values :
                                                     update_values | update_gradients);

        std::unique_ptr<FEPointEvaluation<dim, dim>> normal_vals;
        std::unique_ptr<FESystem<dim>>               fe_normal;
        if (normal_vector)
          {
            fe_normal   = std::make_unique<FESystem<dim>>(scratch_data.get_fe(normal_dof_idx), dim);
            normal_vals = std::make_unique<FEPointEvaluation<dim, dim>>(scratch_data.get_mapping(),
                                                                        *fe_normal,
                                                                        update_values);
          }

        FEPointEvaluation<1, dim> heat_source_vals(scratch_data.get_mapping(),
                                                   scratch_data.get_fe(temp_dof_idx),
                                                   update_values);

        std::vector<double>                  buffer;
        std::vector<double>                  buffer_dim;
        std::vector<types::global_dof_index> local_dof_indices;

        LevelSet::Tools::evaluate_at_interface<dim>(
          scratch_data.get_dof_handler(temp_dof_idx),
          scratch_data.get_mapping(),
          level_set_heaviside,
          [&](const auto &cell, const auto &points_real, const auto &points, const auto &weights) {
            (void)points_real;
            // evaluate rhs term
            local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
            buffer.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            const unsigned int n_points = points.size();

            const ArrayView<const Point<dim>> unit_points(points.data(), n_points);
            const ArrayView<const double>     JxW(weights.data(), n_points);

            ls.reinit(cell, unit_points);
            heat_source_vals.reinit(cell, unit_points);

            // gather evaluate level set for the points at the interface
            ls.reinit(cell, unit_points);
            scratch_data.get_constraint(ls_dof_idx)
              .get_dof_values(level_set_heaviside,
                              local_dof_indices.begin(),
                              buffer.begin(),
                              buffer.end());
            ls.evaluate(buffer,
                        normal_vector ? EvaluationFlags::values : EvaluationFlags::gradients);

            // gather_evaluate unit normal vector for the points at the interface
            if (normal_vals && normal_vector)
              {
                normal_vals->reinit(cell, unit_points);
                buffer_dim.resize(fe_normal->n_dofs_per_cell()); // @todo: times dim

                for (int d = 0; d < dim; ++d)
                  {
                    cell->get_dof_values(normal_vector->block(d), buffer.begin(), buffer.end());

                    for (unsigned int c = 0;
                         c < scratch_data.get_fe(normal_dof_idx).n_dofs_per_cell();
                         ++c)
                      buffer_dim[fe_normal->component_to_system_index(d, c)] = buffer[c];
                  }

                // normalize
                for (unsigned int c = 0; c < cell->get_fe().n_dofs_per_cell(); ++c)
                  {
                    double norm = 0.0;
                    for (int d = 0; d < dim; ++d)
                      norm += std::pow(buffer_dim[fe_normal->component_to_system_index(d, c)], 2);

                    norm = std::max(1e-6, std::sqrt(norm));

                    for (int d = 0; d < dim; ++d)
                      buffer_dim[fe_normal->component_to_system_index(d, c)] /= norm;
                  }
                normal_vals->evaluate(make_array_view(buffer_dim), EvaluationFlags::values);
              }

            for (unsigned int q = 0; q < n_points; ++q)
              {
                // If a normal vector field is given, use it to compute the unit normal to the
                // interface. Otherwise, use the gradient of the level set field.
                const auto unit_normal =
                  normal_vector ? normal_vals->get_value(q) :
                                  ls.get_gradient(q) / std::max(ls.get_gradient(q).norm(), 1e-6);
                const auto result =
                  local_compute_interfacial_heat_source(laser_power_density, unit_normal, 1.0) *
                  JxW[q];

                heat_source_vals.submit_value(result, q);
              }

            // integrate laser heat source
            heat_source_vals.integrate(buffer, EvaluationFlags::values);

            scratch_data.get_constraint(temp_dof_idx)
              .distribute_local_to_global(buffer, local_dof_indices, heat_rhs);
          },
          0.5, /*contour value*/
          3 /*n_subdivisions*/);

        heat_rhs.compress(VectorOperation::add);

        heat_rhs.zero_out_ghost_values();
        level_set_heaviside.zero_out_ghost_values();
        if (normal_vector)
          normal_vector->zero_out_ghost_values();
      }
    else
      {
        Assert(false, ExcNotImplemented());
        (void)heat_rhs;
        (void)scratch_data;
        (void)temp_dof_idx;
        (void)laser_power_density;
        (void)laser_position;
        (void)level_set_heaviside;
        (void)ls_dof_idx;
        (void)zero_out;
        (void)normal_vector;
        (void)normal_dof_idx;
      }
  }

  template class LaserHeatSourceUniform<1>;
  template class LaserHeatSourceUniform<2>;
  template class LaserHeatSourceUniform<3>;
} // namespace MeltPoolDG::Heat
