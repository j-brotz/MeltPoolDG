#include <deal.II/base/exceptions.h>

#include <meltpooldg/utilities/profiling_data.hpp>

namespace MeltPoolDG::Profiling
{
  template <typename number>
  void
  ProfilingData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("profiling");
    {
      prm.add_parameter(
        "enable wall time statistics",
        enable_wall_time_statistics,
        "Set this parameter to true if profiling of wall time should be enabled. It will be automatically"
        "enabled for verbosity level >=1.");
      prm.add_parameter(
        "enable iteration statistics",
        enable_iteration_statistics,
        "Set this parameter to true if profiling of iteration statistics should be enabled. It will be automatically"
        "enabled for verbosity level >=1.");
      prm.add_parameter(
        "enable dof statistics",
        enable_dof_statistics,
        "Set this parameter to true if profiling of DoF statistics should be enabled. It will be automatically"
        "enabled for verbosity level >=1.");
      prm.add_parameter(
        "enable cell statistics",
        enable_cell_statistics,
        "Set this parameter to true if profiling of cell statistics should be enabled. It will be automatically"
        "enabled for verbosity level >=1.");
      prm.add_parameter("write time step size",
                        write_time_step_size,
                        "Write profiling output every given time step size. If this parameter is "
                        "set, the specified parameter for write frequency is overwritten.");
      prm.add_parameter("time type",
                        time_type,
                        "Choose the type of time measure to write profiling information.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  ProfilingData<number>::check_input_parameters(const number time_step_size) const
  {
    /*
     * calculate the profiling output frequency if a time step size
     */
    if (write_time_step_size > 0.0)
      AssertThrow(time_type == TimeType::real || write_time_step_size >= time_step_size,
                  dealii::ExcMessage(
                    "The time step size for profiling must be equal or larger than the simulation "
                    "time step size."));

    if (enable_wall_time_statistics or enable_iteration_statistics or enable_dof_statistics or
        enable_cell_statistics)
      enable = true;
  }

  template struct ProfilingData<double>;
} // namespace MeltPoolDG::Profiling