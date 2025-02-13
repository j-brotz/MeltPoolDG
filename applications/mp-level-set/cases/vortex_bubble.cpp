#include "vortex_bubble.hpp"

namespace MeltPoolDG::Simulation::VortexBubble
{
  // Explicit instantiations for the required dimensions
  template class SimulationVortexBubble<1>;
  template class SimulationVortexBubble<2>;
  template class SimulationVortexBubble<3>;
} // namespace MeltPoolDG::Simulation::VortexBubble
