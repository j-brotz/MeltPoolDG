#include "../../mp-radiative-transport/cases/radiative_transport.templates.hpp"
//
#include "../heat_transfer_case.hpp"

namespace MeltPoolDG::Simulation::RadiativeTransport
{
  template class SimulationRadTrans<1, Heat::HeatTransferCase<1>>;
  template class SimulationRadTrans<2, Heat::HeatTransferCase<2>>;
  template class SimulationRadTrans<3, Heat::HeatTransferCase<3>>;
} // namespace MeltPoolDG::Simulation::RadiativeTransport
