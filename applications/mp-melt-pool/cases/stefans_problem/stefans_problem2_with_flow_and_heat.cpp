#include "stefans_problem2_with_flow_and_heat.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::StefansProblem2WithFlowAndHeat
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationStefansProblem2WithFlowAndHeat,
                           "stefans_problem2_with_flow_and_heat",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationStefansProblem2WithFlowAndHeat,
                           "stefans_problem2_with_flow_and_heat",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationStefansProblem2WithFlowAndHeat,
                           "stefans_problem2_with_flow_and_heat",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::StefansProblem2WithFlowAndHeat
