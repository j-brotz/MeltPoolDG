#include "channel_particle_flow.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  // Explicit instantiations for the required dimensions
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationChannelParticleFlow,
                           "channel_particle_flow",
                           2);
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationChannelParticleFlow,
                           "channel_particle_flow",
                           3);
} // namespace MeltPoolDG::Simulation::CompressibleFlow
