#include "static_two_phase.hpp"

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  // Explicit instantiations for the required dimensions
  template class SimulationStaticTwoPhase<1>;
  template class SimulationStaticTwoPhase<2>;
  template class SimulationStaticTwoPhase<3>;
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase