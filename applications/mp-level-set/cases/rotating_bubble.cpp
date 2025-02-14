#include "rotating_bubble.hpp"

namespace MeltPoolDG::Simulation::RotatingBubble
{
  // Explicit instantiations for the required dimensions
  template class SimulationRotatingBubble<1>;
  template class SimulationRotatingBubble<2>;
  template class SimulationRotatingBubble<3>;
} // namespace MeltPoolDG::Simulation::RotatingBubble
