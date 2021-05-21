#include <meltpooldg/heat/laser.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserOperation<dim>::LaserOperation(const ScratchData<dim> &    scratch_data_in,
                                      const LaserData<double> &   laser_data_in,
                                      const MaterialData<double> &material_data_in)
    : scratch_data(scratch_data_in)
    , laser_data(laser_data_in)
    , material(material_data_in)
    , laser_position(
        MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(laser_data.center))
  {}

  template <int dim>
  void
  LaserOperation<dim>::set_initial_condition(const double start_time)
  {
    current_time = start_time;
    compute_laser_intensity();
    scratch_data.get_pcout() << "current laser position: " << laser_position << std::endl;
    scratch_data.get_pcout() << "current laser intensity: " << laser_intensity << std::endl;
  }

  template <int dim>
  void
  LaserOperation<dim>::move_laser(const double dt)
  {
    // 0) update current time
    current_time += dt;
    // 1) compute the current center of the laser beam
    if (laser_data.do_move)
      laser_position[0] += laser_data.scan_speed * dt;
    // 2) update intensity of the laser
    compute_laser_intensity();

    scratch_data.get_pcout() << "current laser position: " << laser_position << std::endl;
    scratch_data.get_pcout() << "current laser intensity: " << laser_intensity << std::endl;
  }

  template <int dim>
  void
  LaserOperation<dim>::compute_analytical_temperature_field(
    const VectorType &          level_set_as_heaviside,
    VectorType &                temperature,
    const unsigned int          temp_dof_idx,
    const double                density_gas,
    const double                density_liquid,
    const MeltPoolData<double> &mp_data,
    const double                level_set_value_is_gas)
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
              analytical_temperature_field(support_points[local_dof_indices[i]],
                                           level_set_as_heaviside[local_dof_indices[i]],
                                           mp_data,
                                           density_gas,
                                           density_liquid,
                                           level_set_value_is_gas);
        }

    temperature.compress(VectorOperation::insert);
    scratch_data.get_constraint(temp_dof_idx).distribute(temperature);
    level_set_as_heaviside.zero_out_ghosts();
  }

  template <int dim>
  Point<dim>
  LaserOperation<dim>::get_laser_position()
  {
    return laser_position;
  }

  template <int dim>
  double
  LaserOperation<dim>::get_laser_power()
  {
    return laser_intensity * laser_data.power;
  }

  template <int dim>
  void
  LaserOperation<dim>::compute_laser_intensity()
  {
    if (laser_data.power_over_time == "ramp")
      {
        AssertThrow(
          laser_data.power_end_time > laser_data.power_start_time,
          ExcMessage(
            "For the temporal ramp distribution of the laser power,"
            " the parameter laser power end time must be larger than laser power start time."));
        laser_intensity = (current_time - laser_data.power_start_time) /
                          (laser_data.power_end_time - laser_data.power_start_time);
        laser_intensity = std::min(std::max(0.0, laser_intensity), 1.0);
      }
    else if (laser_data.power_over_time == "constant")
      {
        if (current_time >= laser_data.power_end_time)
          {
            laser_intensity = 0.0;
          }
        else
          laser_intensity = 1.0;
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  template <int dim>
  double
  LaserOperation<dim>::analytical_temperature_field(Point<dim>                  point,
                                                    const double                heaviside,
                                                    const MeltPoolData<double> &mp_data,
                                                    const double                density_gas,
                                                    const double                density_liquid,
                                                    const double level_set_value_is_gas)
  {
    const double  P  = laser_data.power * laser_intensity;
    const double &v  = laser_data.scan_speed;
    const double &T0 = mp_data.ambient_temperature;

    double weight;

    if (laser_data.variable_properties_over_interface)
      weight = (level_set_value_is_gas == -1) ? heaviside : (1. - heaviside);
    else
      {
        double sharp_heaviside = (heaviside > 0.5) ? 1.0 : 0.0;
        weight = (level_set_value_is_gas == -1) ? sharp_heaviside : (1. - sharp_heaviside);
      }

    const double absorptivity =
      UtilityFunctions::interpolate(weight, mp_data.gas.absorptivity, mp_data.liquid.absorptivity);
    const double conductivity = UtilityFunctions::interpolate(weight,
                                                              material.first.conductivity,
                                                              material.second.conductivity);
    const double capacity =
      UtilityFunctions::interpolate(weight, material.first.capacity, material.second.capacity);

    const double density             = density_gas + weight * (density_liquid - density_gas);
    const double thermal_diffusivity = conductivity / (density * capacity);

    // modify temperature profile to be anisotropic
    for (int d = 0; d < dim - 1; d++)
      point[d] *= mp_data.temperature_x_to_y_ratio;

    double R = point.distance(laser_position);

    if (R == 0.0)
      R = 1e-16;

    double T = P * absorptivity / (4 * numbers::PI * R * conductivity) *
                 std::exp(-v * R / (2. * thermal_diffusivity)) +
               T0;

    return (T > mp_data.max_temperature) ? mp_data.max_temperature : T;
  }

  template class LaserOperation<1>;
  template class LaserOperation<2>;
  template class LaserOperation<3>;
} // namespace MeltPoolDG::Heat
