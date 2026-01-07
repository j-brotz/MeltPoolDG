#pragma once

#include <deal.II/base/parameter_handler.h>


namespace MeltPoolDG
{
  template <typename number>
  struct ObstacleData
  {
    /// If true, obstacles are fixed and do not move during the simulation.
    bool stationary_obstacles = false;

    /// Path to the input file containing the initial obstacle state.
    std::string obstacle_state_input_file;

    /// Gravity constant
    number gravity_constant = 9.81;

    struct
    {
      /// If true and AMR is enabled, regions around obstacles are adaptively refined according to
      /// the specified refinement parameters.
      bool do_refine_obstacles = false;

      /// Fraction of a reference length (e.g., particle radius) used to determine the  distance
      /// from the obstacle surface within which cells inside the obstacles are refined.
      number inner_fractional_distance_to_surface = 1.;

      /// Fraction of a reference length (e.g., particle radius) used to determine the  distance
      /// from the obstacle surface within which cells outside the obstacles are refined.
      number outer_fractional_distance_to_surface = 1.;
    } amr;

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
        prm.add_parameter("gravity constant", gravity_constant);
        prm.enter_subsection("amr");
        {
          prm.add_parameter(
            "do refine",
            amr.do_refine_obstacles,
            "If true and AMR is enabled, regions around obstacles are adaptively refined "
            "according to the specified refinement parameters.");
          prm.add_parameter(
            "inner surface distance",
            amr.inner_fractional_distance_to_surface,
            "Fraction of a reference length (e.g., particle radius) used to determine the "
            "distance from the obstacle surface within which cells inside the obstacles "
            "are refined.");
          prm.add_parameter(
            "outer surface distance",
            amr.outer_fractional_distance_to_surface,
            "Fraction of a reference length (e.g., particle radius) used to determine the "
            "distance from the obstacle surface within which cells outside the obstacles "
            "are refined.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
