#include "rotating_bubble.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::RotatingBubble
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationRotatingBubble,
                           "rotating_bubble",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationRotatingBubble,
                           "rotating_bubble",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationRotatingBubble,
                           "rotating_bubble",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::RotatingBubble
