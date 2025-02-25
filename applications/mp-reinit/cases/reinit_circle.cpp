#include "reinit_circle.hpp"

#include <meltpooldg/case_registration.hpp>

namespace MeltPoolDG::Simulation::ReinitCircle
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase, SimulationReinit, "reinit_circle", 1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase, SimulationReinit, "reinit_circle", 2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase, SimulationReinit, "reinit_circle", 3);
} // namespace MeltPoolDG::Simulation::ReinitCircle
