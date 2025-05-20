#include "channel_particle_flow.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CfdDem
{
  // Explicit instantiations for the required dimensions and floating point number type
  MELTPOOLDG_REGISTER_CASE(CfdDemCase,
                           SimulationChannelParticleFlow,
                           "channel_particle_flow",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(CfdDemCase,
                           SimulationChannelParticleFlow,
                           "channel_particle_flow",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::CfdDem