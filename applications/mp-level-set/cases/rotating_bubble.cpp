#include "rotating_bubble.hpp"

#include <meltpooldg/case_registration.hpp>

namespace MeltPoolDG::Simulation::RotatingBubble
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationRotatingBubble, "rotating_bubble", 1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationRotatingBubble, "rotating_bubble", 2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationRotatingBubble, "rotating_bubble", 3);
} // namespace MeltPoolDG::Simulation::RotatingBubble
