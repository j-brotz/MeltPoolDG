#include "evaporating_droplet.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::EvaporatingDroplet
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingDroplet,
                           "evaporating_droplet",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingDroplet,
                           "evaporating_droplet",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingDroplet,
                           "evaporating_droplet",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::EvaporatingDroplet
