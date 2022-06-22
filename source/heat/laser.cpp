#include <meltpooldg/heat/laser.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Heat
{
  LaserImpactType
  impact_type_to_emun(const std::string &specifier)
  {
    if (specifier == "volumetric")
      return LaserImpactType::volumetric;
    else if (specifier == "interface")
      return LaserImpactType::interface;
    else
      AssertThrow(false, ExcMessage("Unknown laser impact type! Abort..."));
  }

  namespace
  {
    template <int dim, typename ForwardIterator>
    Point<dim>
    to_point(const ForwardIterator begin, const ForwardIterator end)
    {
      (void)end;
      AssertIndexRange(dim, std::distance(begin, end) + 1);

      Point<dim> point;

      auto it = begin;
      for (int i = 0; i < dim; ++i)
        point[i] = *it++;

      return point;
    }
  } // namespace

  template <int dim>
  LaserOperation<dim>::LaserOperation(const ScratchData<dim> &    scratch_data_in,
                                      const LaserData<double> &   laser_data_in,
                                      const MaterialData<double> &material_data_in)
    : scratch_data(scratch_data_in)
    , laser_data(laser_data_in)
    , material(material_data_in)
    , laser_position(to_point<dim>(laser_data.center.begin(), laser_data.center.end()))
    , impact_type(impact_type_to_emun(laser_data.impact_type))
  {}

  template <int dim>
  void
  LaserOperation<dim>::set_initial_condition(const double start_time)
  {
    current_time = start_time;
    compute_laser_intensity();
    std::ostringstream str;
    str << "current laser position: " << laser_position;
    Journal::print_line(scratch_data.get_pcout(), str.str(), "laser");
    str.str("");
    str << "current laser intensity: " << laser_intensity;
    Journal::print_line(scratch_data.get_pcout(), str.str(), "laser");
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

    std::ostringstream str;
    str << "current laser position: " << laser_position;
    Journal::print_line(scratch_data.get_pcout(), str.str(), "laser");
    str.str("");
    str << "current laser intensity: " << laser_intensity;
    Journal::print_line(scratch_data.get_pcout(), str.str(), "laser");
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
  LaserImpactType
  LaserOperation<dim>::get_laser_impact_type() const
  {
    return impact_type;
  }

  template class LaserOperation<1>;
  template class LaserOperation<2>;
  template class LaserOperation<3>;
} // namespace MeltPoolDG::Heat
