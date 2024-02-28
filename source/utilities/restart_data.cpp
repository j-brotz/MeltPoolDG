#include <deal.II/base/exceptions.h>

#include <meltpooldg/utilities/restart_data.hpp>

#include <filesystem>

namespace MeltPoolDG::Restart
{
  template <typename number>
  void
  RestartData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("restart");
    {
      prm.add_parameter(
        "save",
        save,
        "Set this parameter to any number >= 0 to specify how many restart files should be kept. "
        "-1 means no restart save.");
      prm.add_parameter(
        "load",
        load,
        "Set this parameter to any number >= 0 to specify which restart file should be loaded. "
        "-1 means no restart load.");
      prm.add_parameter("write time step size",
                        write_time_step_size,
                        "Write restart output every given time step size. If this parameter is "
                        "set, the specified parameter for write frequency is overwritten.");
      prm.add_parameter("time type", time_type, "Choose the type of time measure to write ");
      prm.add_parameter("directory", directory, "Write restart directory");
      prm.add_parameter("prefix", prefix, "Write restart prefix");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  RestartData<number>::post(const std::string &output_directory)
  {
    /*
     * calculate the restart output frequency if a time step size
     */
    if (write_time_step_size <= 0.0)
      {
        time_type            = TimeType::real;
        write_time_step_size = 3600; // 1 hour
      }
    /*
     * set default value of restart prefix
     */
    if (directory == "")
      directory = std::filesystem::path(std::filesystem::current_path()) /
                  std::filesystem::path(output_directory);
    // modify the value of prefix for easy internal access
    prefix = std::filesystem::path(directory) / std::filesystem::path(prefix);
  }

  template <typename number>
  void
  RestartData<number>::check_input_parameters(const number time_step_size) const
  {
    if (write_time_step_size > 0.0)
      {
        AssertThrow(time_type == TimeType::real || write_time_step_size >= time_step_size,
                    ExcMessage(
                      "The time step size for restart must be equal or larger than the simulation "
                      "time step size."));
      }
  }

  template struct RestartData<double>;
} // namespace MeltPoolDG::Restart