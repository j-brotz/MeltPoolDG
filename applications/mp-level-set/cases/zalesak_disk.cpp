#include "zalesak_disk.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::ZalesakDisk
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationZalesakDisk,
                           "zalesak_disk",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationZalesakDisk,
                           "zalesak_disk",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::LevelSetCase,
                           SimulationZalesakDisk,
                           "zalesak_disk",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::ZalesakDisk
