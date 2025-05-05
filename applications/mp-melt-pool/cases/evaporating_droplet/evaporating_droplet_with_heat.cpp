#include "evaporating_droplet_with_heat.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::EvaporatingDropletWithHeat
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingDropletWithHeat,
                           "evaporating_droplet_with_heat",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingDropletWithHeat,
                           "evaporating_droplet_with_heat",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationEvaporatingDropletWithHeat,
                           "evaporating_droplet_with_heat",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::EvaporatingDropletWithHeat
