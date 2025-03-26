#include "static_two_phase.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  // Explicit instantiations for the required dimensions
  MELTPOOLDG_REGISTER_CASE(Multiphase::CompressibleMultiphaseCase,
                           SimulationStaticTwoPhase,
                           "static_two_phase",
                           1);
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase