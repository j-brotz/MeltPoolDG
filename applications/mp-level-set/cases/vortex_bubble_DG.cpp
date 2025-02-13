#include "vortex_bubble_DG.hpp"

namespace MeltPoolDG::Simulation::VortexBubbleDG
{
  // Explicit instantiations for the required dimensions
  template class SimulationVortexBubbleDG<1>;
  template class SimulationVortexBubbleDG<2>;
  template class SimulationVortexBubbleDG<3>;
} // namespace MeltPoolDG::Simulation::VortexBubbleDG
