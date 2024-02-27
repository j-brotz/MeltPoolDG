#include <meltpooldg/utilities/time_stepping_data.hpp>


namespace MeltPoolDG
{
  template <typename number>
  void
  TimeSteppingData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("time stepping");
    {
      prm.add_parameter("start time",
                        start_time,
                        "Defines the start time for the solution of the levelset problem");
      prm.add_parameter("end time",
                        end_time,
                        "Sets the end time for the solution of the levelset problem");
      prm.add_parameter("time step size",
                        time_step_size,
                        "Sets the step size for time stepping. For non-uniform "
                        "time stepping, this parameter determines the size of the first "
                        "time step.");
      prm.add_parameter("max n steps", max_n_steps, "Sets the maximum number of melt_pool steps");
      prm.add_parameter("time step size function",
                        time_step_size_function,
                        "Set an analytical function to determine the time step size. "
                        "For the prediction of the new time increment, the old time is used.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  TimeStepLimitData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("time step limit");
    {
      prm.add_parameter(
        "enable",
        enable,
        "Set this parameter to true to check whether the time step limit is not exceeded.");
      prm.add_parameter("scale factor",
                        scale_factor,
                        "Scale factor between 0 and 1 to compute the time step limit.");
    }
    prm.leave_subsection();
  }

  template struct TimeSteppingData<double>;
  template struct TimeStepLimitData<double>;
} // namespace MeltPoolDG