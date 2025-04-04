#include "radiative_transport.templates.hpp"
//
#include "../radiative_transport_case.hpp"


namespace MeltPoolDG::Simulation::RadiativeTransport
{
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::RadiativeTransport::RadiativeTransportCase,
                                     SimulationRadTrans,
                                     "radiative_transport",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::RadiativeTransport::RadiativeTransportCase,
                                     SimulationRadTrans,
                                     "radiative_transport",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::RadiativeTransport::RadiativeTransportCase,
                                     SimulationRadTrans,
                                     "radiative_transport",
                                     3,
                                     double);
} // namespace MeltPoolDG::Simulation::RadiativeTransport
