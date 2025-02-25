#include "vortex_bubble_DG.hpp"

#include <meltpooldg/case_registration.hpp>

namespace MeltPoolDG::Simulation::VortexBubbleDG
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationVortexBubbleDG, "vortex_bubble_DG", 1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationVortexBubbleDG, "vortex_bubble_DG", 2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationVortexBubbleDG, "vortex_bubble_DG", 3);
} // namespace MeltPoolDG::Simulation::VortexBubbleDG
