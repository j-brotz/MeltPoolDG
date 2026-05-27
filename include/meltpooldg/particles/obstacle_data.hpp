#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/particles/cohesive_forces_data.hpp>
#include <meltpooldg/particles/contact_forces_data.hpp>


namespace MeltPoolDG
{
  template <typename number>
  struct ObstacleDataStructureData
  {
    number max_sphere_of_influence_radius = std::numeric_limits<number>::max();

    number skin_thickness = 0.0;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("obstacle data structure");
      {
        prm.add_parameter(
          "radius max sphere of influence",
          max_sphere_of_influence_radius,
          "Maximum radius of influence for obstacles. This parameter is used to determine the maximum distance at which an obstacle can affect another solution. It is used to determine the optimal cell level on which to cache the particles for efficient search.");
        prm.add_parameter(
          "skin thickness",
          skin_thickness,
          "Thickness of the skin layer used in the dynamic update control. This layer defines a buffer which is added to the minimum cell size for the cells in which the particles are cached such that the cache does not need to be updated every time a particle moves. Instead, the cache is only updated when a particle moves outside of the skin layer. The optimal value for this parameter depends on the expected particle velocities and the time step size, and should be chosen such that particles do not frequently move outside of the skin layer, which would trigger frequent updates of the cache and thus reduce performance. On the other hand, choosing a very large skin thickness can lead to increased computational cost during the search for relevant particles, since more particles will be included in the search than necessary. Therefore, it is important to find a balance when choosing this parameter.");
      }
      prm.leave_subsection();
    }
  };

  template <typename number>
  struct ObstacleData
  {
    /// If true, obstacles are fixed and do not move during the simulation.
    bool stationary_obstacles = false;

    /// Path to the input file containing the initial obstacle state.
    std::string obstacle_state_input_file;

    /// Data structure parameters for the obstacle data.
    ObstacleDataStructureData<number> data_structure_data;

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

    /// Cohesive force data defining the material and cohesive contact properties for all particles.
    SphericalParticleCohesiveForceData<number> cohesive_forces;

    /// Contact force data defining the material and contact properties for all particles.
    SphericalParticleContactData<number> contact_forces;

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

        data_structure_data.add_parameters(prm);
        cohesive_forces.add_parameters(prm);
        contact_forces.add_parameters(prm);
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
