#include "reinit_circle_DG.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::ReinitCircleDG
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinitDG,
                           "reinit_circle_DG",
                           1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinitDG,
                           "reinit_circle_DG",
                           2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinitDG,
                           "reinit_circle_DG",
                           3);
} // namespace MeltPoolDG::Simulation::ReinitCircleDG
