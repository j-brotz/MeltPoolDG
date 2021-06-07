#include <meltpooldg/heat/laser_heat_source_gauss.hpp>
//

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserHeatSourceGauss<dim>::LaserHeatSourceGauss(const LaserData<double>::GaussData &data_in)
    : data(data_in)
    , vol_peak_power_density_factor(
        1. / std::pow(data.laser_beam_radius * std::sqrt(numbers::PI / 2), 3))
    , surf_peak_power_density_factor(
        1. / (data.laser_beam_radius * data.laser_beam_radius * numbers::PI / 2))
  {
    AssertThrow(data.laser_beam_radius > 0.0,
                ExcMessage("The laser beam radius must be greater than zero! Abort.."));
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::local_compute_volumetric_heat_source(const Point<dim> &position,
                                                                  const Point<dim> &laser_position,
                                                                  const double      power) const
  {
    const double distance = position.distance(laser_position);
    return data.absorptivity * power_density_volumetric(distance, power);
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::local_compute_interfacial_heat_source(
    const Point<dim> &            position,
    const Point<dim> &            laser_position,
    double                        power,
    const Tensor<1, dim, double> &normal_vector,
    double                        delta_value) const
  {
    // only consider distance in x (2D) or x and y (3D) direction. Disregard distance in dim-1
    // direction.
    Point<dim - 1> distance_vector;
    distance_vector[0] = position[0] - laser_position[0];
    if constexpr (dim == 3)
      distance_vector[1] = position[1] - laser_position[1];
    const double distance = distance_vector.norm();

    // assume laser direction coincides with the negative dim-1 direction
    double projection_factor = normal_vector * -Point<dim>::unit_vector(dim - 1);
    if (projection_factor < 0.0)
      projection_factor = 0.0;

    return data.absorptivity * projection_factor * delta_value *
           power_density_interfacial(distance, power);
  }

  template <int dim>
  void
  LaserHeatSourceGauss<dim>::compute_interfacial_heat_source(VectorType &heat_source_vector,
                                                             const ScratchData<dim> &scratch_data,
                                                             const unsigned int      temp_dof_idx,
                                                             const double            laser_power,
                                                             const Point<dim> &      laser_position,
                                                             const VectorType & level_set_heaviside,
                                                             const unsigned int ls_dof_idx,
                                                             const bool         zero_out) const
  {
    if (zero_out)
      scratch_data.initialize_dof_vector(heat_source_vector, temp_dof_idx);

    FEValues<dim> heat_source_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(temp_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(temp_dof_idx).get_fe().get_unit_support_points()),
      update_quadrature_points);

    level_set_heaviside.update_ghost_values();
    FEValues<dim> ls_heaviside_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(temp_dof_idx).get_fe().get_unit_support_points()),
      update_gradients);

    const unsigned int dofs_per_cell =
      scratch_data.get_dof_handler(temp_dof_idx).get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    // TODO: find a better way to interpolate the (level-set dependent) laser heat source onto the
    // heat source dof vector. Problem: if ls-degree=1, its gradient may be discontinuous!
    for (const auto &cell : scratch_data.get_dof_handler(temp_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            heat_source_eval.reinit(cell);
            ls_heaviside_eval.reinit(cell);

            std::vector<dealii::Tensor<1, dim, double>> grad_ls_heaviside(
              ls_heaviside_eval.n_quadrature_points);
            ls_heaviside_eval.get_function_gradients(level_set_heaviside, grad_ls_heaviside);

            for (const auto q : heat_source_eval.quadrature_point_indices())
              {
                const double delta_value = grad_ls_heaviside[q].norm();
                if (delta_value == 0.0)
                  {
                    heat_source_vector[local_dof_indices[q]] = 0.0;
                    continue;
                  }
                // TODO: use computed normal vector
                const Tensor<1, dim, double> normal_vector = grad_ls_heaviside[q] / delta_value;

                const double temp =
                  local_compute_interfacial_heat_source(heat_source_eval.quadrature_point(q),
                                                        laser_position,
                                                        laser_power,
                                                        normal_vector,
                                                        delta_value);
                heat_source_vector[local_dof_indices[q]] = temp;
              }
          }
      }
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::power_density_volumetric(const double radius, const double power) const
  {
    const double s          = radius / data.laser_beam_radius;
    const double peak_power = power * vol_peak_power_density_factor;
    return peak_power * std::exp(-2. * s * s);
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::power_density_interfacial(double radius, double power) const
  {
    const double s          = radius / data.laser_beam_radius;
    const double peak_power = power * surf_peak_power_density_factor;
    return peak_power * std::exp(-2. * s * s);
  }

  template class LaserHeatSourceGauss<1>;
  template class LaserHeatSourceGauss<2>;
  template class LaserHeatSourceGauss<3>;
} // namespace MeltPoolDG::Heat