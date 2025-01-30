#include "radiative_transport.hpp"

namespace MeltPoolDG::Simulation::RadiativeTransport
{
  // Explicit instantiations for the required dimensions
  template class SimulationRadTrans<1>;
  template class SimulationRadTrans<2>;
  template class SimulationRadTrans<3>;
} // namespace MeltPoolDG::Simulation::RadiativeTransport
