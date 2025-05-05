#include "oscillating_droplet.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::OscillatingDroplet
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationOscillatingDroplet,
                           "oscillating_droplet",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationOscillatingDroplet,
                           "oscillating_droplet",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationOscillatingDroplet,
                           "oscillating_droplet",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::OscillatingDroplet
