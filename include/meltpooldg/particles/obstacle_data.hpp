#pragma once

#include <deal.II/base/parameter_handler.h>

struct ObstacleData
{
  /// If true, obstacles are fixed and do not move during the simulation.
  bool stationary_obstacles = false;

  /// Path to the input file containing the initial obstacle state.
  std::string obstacle_state_file;

  void
  add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Obstacles");
    {
      prm.add_parameter("stationary",
                        stationary_obstacles,
                        "Set to true if obstacles shall be stationary.");
      prm.add_parameter("obstacle state file",
                        obstacle_state_file,
                        "File in which the obstacle state data is stored.");
    }
    prm.leave_subsection();
  }
};
