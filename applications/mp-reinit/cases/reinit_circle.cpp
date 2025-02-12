#include "reinit_circle.hpp"

namespace MeltPoolDG::Simulation::ReinitCircle
{
  // Explicit instantiations for the required dimensions
  template class SimulationReinit<1>;
  template class SimulationReinit<2>;
  template class SimulationReinit<3>;
} // namespace MeltPoolDG::Simulation::ReinitCircle
