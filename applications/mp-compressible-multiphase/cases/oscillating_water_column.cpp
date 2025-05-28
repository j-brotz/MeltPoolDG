#include "oscillating_water_column.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  // Explicit instantiations for the required dimensions
  MELTPOOLDG_REGISTER_CASE(Multiphase::CompressibleMultiphaseCase,
                           SimulationOscillatingWaterColumn,
                           "oscillating_water_column",
                           1,
                           double);
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase
