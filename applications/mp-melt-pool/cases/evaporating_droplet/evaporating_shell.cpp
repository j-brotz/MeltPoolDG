#include "evaporating_shell.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::EvaporatingShell
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingShell,
                           "evaporating_shell",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingShell,
                           "evaporating_shell",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingShell,
                           "evaporating_shell",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::EvaporatingShell
