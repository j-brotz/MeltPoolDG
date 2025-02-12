#include "radiative_transport.templates.hpp"
//
#include "../radiative_transport_case.hpp"


namespace MeltPoolDG::Simulation::RadiativeTransport
{
  // Explicit instantiations for the required dimensions
  template class SimulationRadTrans<1, MeltPoolDG::RadiativeTransport::RadiativeTransportCase<1>>;
  template class SimulationRadTrans<2, MeltPoolDG::RadiativeTransport::RadiativeTransportCase<2>>;
  template class SimulationRadTrans<3, MeltPoolDG::RadiativeTransport::RadiativeTransportCase<3>>;
} // namespace MeltPoolDG::Simulation::RadiativeTransport
