#include "channel_particle_flow.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  // Explicit instantiations for the required dimensions
  template class SimulationChannelParticleFlow<2>;
  template class SimulationChannelParticleFlow<3>;
} // namespace MeltPoolDG::Simulation::CompressibleFlow