#include <meltpooldg/heat/laser_analytical_temperature_field.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserAnalyticalTemperatureField<dim>::LaserAnalyticalTemperatureField(
    const LaserData<double>::AnalyticalData &data_in,
    const MaterialData<double> &             material_in,
    const double                             scan_speed_in)
    : laser_data(data_in)
    , material(material_in)
    , scan_speed(scan_speed_in)
  {}

  template <int dim>
  void
  LaserAnalyticalTemperatureField<dim>::compute_temperature_field(
    const ScratchData<dim> &scratch_data,
    const VectorType &      level_set_as_heaviside,
    VectorType &            temperature,
    const unsigned int      temp_dof_idx,
    const double            laser_power,
    const Point<dim> &      laser_position) const
  {
    level_set_as_heaviside.update_ghost_values();
    scratch_data.initialize_dof_vector(temperature, temp_dof_idx);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(temp_dof_idx);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(scratch_data.get_mapping(),
                                         scratch_data.get_dof_handler(temp_dof_idx),
                                         support_points);

    for (const auto &cell : scratch_data.get_dof_handler(temp_dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            temperature[local_dof_indices[i]] =
              local_compute_temperature_field(support_points[local_dof_indices[i]],
                                              level_set_as_heaviside[local_dof_indices[i]],
                                              laser_power,
                                              laser_position);
        }

    temperature.compress(VectorOperation::insert);
    scratch_data.get_constraint(temp_dof_idx).distribute(temperature);
    level_set_as_heaviside.zero_out_ghost_values();
  }

  template <int dim>
  double
  LaserAnalyticalTemperatureField<dim>::local_compute_temperature_field(
    Point<dim>        point,
    const double      heaviside,
    const double      laser_power,
    const Point<dim> &laser_position) const
  {
    const double &P  = laser_power;
    const double &v  = scan_speed;
    const double &T0 = laser_data.ambient_temperature;

    double weight;

    if (laser_data.variable_properties_over_interface)
      weight = heaviside;
    else
      weight = (heaviside > 0.5) ? 1.0 : 0.0;

    const double absorptivity = UtilityFunctions::interpolate(weight,
                                                              laser_data.absorptivity_gas,
                                                              laser_data.absorptivity_liquid);

    const double conductivity = UtilityFunctions::interpolate(weight,
                                                              material.first.conductivity,
                                                              material.second.conductivity);
    const double capacity =
      UtilityFunctions::interpolate(weight, material.first.capacity, material.second.capacity);

    const double density =
      material.first.density + weight * (material.second.density - material.first.density);
    const double thermal_diffusivity = conductivity / (density * capacity);

    // modify temperature profile to be anisotropic
    for (int d = 0; d < dim - 1; d++)
      point[d] *= laser_data.temperature_x_to_y_ratio;

    double R = point.distance(laser_position);

    if (R == 0.0)
      R = 1e-16;

    double T = P * absorptivity / (4 * numbers::PI * R * conductivity) *
                 std::exp(-v * R / (2. * thermal_diffusivity)) +
               T0;

    return (T > laser_data.max_temperature) ? laser_data.max_temperature : T;
  }

  template class LaserAnalyticalTemperatureField<1>;
  template class LaserAnalyticalTemperatureField<2>;
  template class LaserAnalyticalTemperatureField<3>;
} // namespace MeltPoolDG::Heat
