#include "stefans_problem.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::StefansProblem
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationStefansProblem,
                           "stefans_problem",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationStefansProblem,
                           "stefans_problem",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationStefansProblem,
                           "stefans_problem",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::StefansProblem
