#include "vortex_bubble.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::VortexBubble
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationVortexBubble,
                           "vortex_bubble",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationVortexBubble,
                           "vortex_bubble",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationVortexBubble,
                           "vortex_bubble",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::VortexBubble
