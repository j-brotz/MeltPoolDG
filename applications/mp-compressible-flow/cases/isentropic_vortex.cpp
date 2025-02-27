#include "isentropic_vortex.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationIsentropicVortex,
                           "isentropic_vortex",
                           2);
} // namespace MeltPoolDG::Simulation::CompressibleFlow
