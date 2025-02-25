#include "radiative_transport.templates.hpp"
//
#include "../radiative_transport_case.hpp"


namespace MeltPoolDG::Simulation::RadiativeTransport
{
  MELTPOOLDG_REGISTER_RTE_CASE(MeltPoolDG::RadiativeTransport::RadiativeTransportCase,
                               SimulationRadTrans,
                               "radiative_transport",
                               1);
  MELTPOOLDG_REGISTER_RTE_CASE(MeltPoolDG::RadiativeTransport::RadiativeTransportCase,
                               SimulationRadTrans,
                               "radiative_transport",
                               2);
  MELTPOOLDG_REGISTER_RTE_CASE(MeltPoolDG::RadiativeTransport::RadiativeTransportCase,
                               SimulationRadTrans,
                               "radiative_transport",
                               3);
} // namespace MeltPoolDG::Simulation::RadiativeTransport
