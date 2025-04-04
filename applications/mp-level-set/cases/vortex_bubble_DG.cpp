#include "vortex_bubble_DG.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::VortexBubbleDG
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationVortexBubbleDG,
                           "vortex_bubble_DG",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationVortexBubbleDG,
                           "vortex_bubble_DG",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationVortexBubbleDG,
                           "vortex_bubble_DG",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::VortexBubbleDG
