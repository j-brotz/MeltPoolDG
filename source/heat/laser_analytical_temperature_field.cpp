#include <meltpooldg/heat/laser_analytical_temperature_field.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  void
  LaserAnalyticalTemperatureField<dim>::compute_temperature_field(
    const ScratchData<dim>     &scratch_data,
    const MaterialData<double> &material,
    const LaserData<double>    &laser_data,
    const double                laser_power,
    const Point<dim>           &laser_position,
    VectorType                 &temperature,
    const VectorType           &level_set_as_heaviside,
    const unsigned int          temp_dof_idx)
  {
    // set the maximum temperature of the melt pool if not specified

    const bool update_ghosts = !level_set_as_heaviside.has_ghost_elements();

    if (update_ghosts)
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
            {
              const double T =
                local_compute_temperature_field(material,
                                                laser_data,
                                                support_points[local_dof_indices[i]],
                                                level_set_as_heaviside[local_dof_indices[i]],
                                                laser_data.scan_speed,
                                                laser_power,
                                                laser_position);
              temperature[local_dof_indices[i]] = (T > laser_data.analytical.max_temperature) ?
                                                    laser_data.analytical.max_temperature :
                                                    T;
            }
        }

    temperature.compress(VectorOperation::insert);
    scratch_data.get_constraint(temp_dof_idx).distribute(temperature);

    // update ghost values of solution
    temperature.update_ghost_values();

    if (update_ghosts)
      level_set_as_heaviside.zero_out_ghost_values();
  }

  template <int dim>
  double
  LaserAnalyticalTemperatureField<dim>::local_compute_temperature_field(
    const MaterialData<double> &material,
    const LaserData<double>    &laser_data,
    const Point<dim>           &point,
    const double                heaviside,
    const double                scan_speed,
    const double                laser_power,
    const Point<dim>           &laser_position)
  {
    const double P  = laser_power;
    const double v  = scan_speed;
    const double T0 = laser_data.analytical.ambient_temperature;

    const double weight = (material.two_phase_fluid_properties_transition_type !=
                           TwoPhaseFluidPropertiesTransitionType::sharp) ?
                            heaviside :
                            ((heaviside > 0.5) ? 1.0 : 0.0);

    const double absorptivity = LevelSet::Tools::interpolate(weight,
                                                             laser_data.absorptivity_gas,
                                                             laser_data.absorptivity_liquid);

    const double conductivity = LevelSet::Tools::interpolate(weight,
                                                             material.gas.thermal_conductivity,
                                                             material.liquid.thermal_conductivity);
    const double capacity     = LevelSet::Tools::interpolate(weight,
                                                         material.gas.specific_heat_capacity,
                                                         material.liquid.specific_heat_capacity);

    const double density =
      material.two_phase_fluid_properties_transition_type ==
          TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation ?
        LevelSet::Tools::interpolate_reciprocal(weight,
                                                material.gas.density,
                                                material.liquid.density) :
        LevelSet::Tools::interpolate(weight, material.gas.density, material.liquid.density);

    const double thermal_diffusivity = conductivity / (density * capacity);

    // modify temperature profile to be anisotropic
    Point<dim> point_scaled = point;

    if (std::abs(laser_data.analytical.temperature_x_to_y_ratio - 1.0) > 1e-10)
      for (int d = 0; d < dim - 1; d++)
        point_scaled[d] *= laser_data.analytical.temperature_x_to_y_ratio;

    double R = point_scaled.distance(laser_position);

    if (R == 0.0)
      R = 1e-16;

    double T = P * absorptivity / (4 * numbers::PI * R * conductivity) *
                 std::exp(-v * R / (2. * thermal_diffusivity)) +
               T0;

    return T;
  }

  template class LaserAnalyticalTemperatureField<1>;
  template class LaserAnalyticalTemperatureField<2>;
  template class LaserAnalyticalTemperatureField<3>;
} // namespace MeltPoolDG::Heat
