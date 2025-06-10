#include "reinit_circle.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::ReinitCircle
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinit,
                           "reinit_circle",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinit,
                           "reinit_circle",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinit,
                           "reinit_circle",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::ReinitCircle
