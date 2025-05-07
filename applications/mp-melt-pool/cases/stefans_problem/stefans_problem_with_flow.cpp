#include "stefans_problem_with_flow.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::StefansProblemWithFlow
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationStefansProblemWithFlow,
                           "stefans_problem_with_flow",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationStefansProblemWithFlow,
                           "stefans_problem_with_flow",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationStefansProblemWithFlow,
                           "stefans_problem_with_flow",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::StefansProblemWithFlow
