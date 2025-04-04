#include "reinit_circle_DG.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::ReinitCircleDG
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinitDG,
                           "reinit_circle_DG",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinitDG,
                           "reinit_circle_DG",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinitDG,
                           "reinit_circle_DG",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::ReinitCircleDG
