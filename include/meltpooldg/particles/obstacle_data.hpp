#pragma once

#include <deal.II/base/parameter_handler.h>


namespace MeltPoolDG
{
  struct ObstacleData
  {
    /// If true, obstacles are fixed and do not move during the simulation.
    bool stationary_obstacles = false;

    /// Path to the input file containing the initial obstacle state.
    std::string obstacle_state_input_file;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("obstacles");
      {
        prm.add_parameter("stationary",
                          stationary_obstacles,
                          "Set to true if obstacles shall be stationary.");
        prm.add_parameter("obstacle state input file",
                          obstacle_state_input_file,
                          "File in which the obstacle initial state data is stored.");
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
