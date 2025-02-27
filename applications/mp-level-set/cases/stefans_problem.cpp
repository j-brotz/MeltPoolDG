#include "stefans_problem.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::StefansProblem
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationStefansProblem, "stefans_problem", 1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationStefansProblem, "stefans_problem", 2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationStefansProblem, "stefans_problem", 3);
} // namespace MeltPoolDG::Simulation::StefansProblem
