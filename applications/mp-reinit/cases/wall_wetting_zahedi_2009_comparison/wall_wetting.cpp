#include "wall_wetting.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::WallWetting
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationWallWetting,
                           "wall_wetting",
                           2,
                           double);
} // namespace MeltPoolDG::Simulation::WallWetting
