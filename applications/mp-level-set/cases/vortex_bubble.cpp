#include "vortex_bubble.hpp"

#include <meltpooldg/case_registration.hpp>

namespace MeltPoolDG::Simulation::VortexBubble
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationVortexBubble, "vortex_bubble", 1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationVortexBubble, "vortex_bubble", 2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationVortexBubble, "vortex_bubble", 3);
} // namespace MeltPoolDG::Simulation::VortexBubble
