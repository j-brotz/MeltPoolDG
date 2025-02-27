#include "zalesak_disk.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::ZalesakDisk
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationZalesakDisk, "zalesak_disk", 1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationZalesakDisk, "zalesak_disk", 2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase, SimulationZalesakDisk, "zalesak_disk", 3);
} // namespace MeltPoolDG::Simulation::ZalesakDisk
