#include "thermo_capillary_droplet.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::ThermoCapillaryDroplet
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationThermoCapillaryDroplet,
                           "thermo_capillary_droplet",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationThermoCapillaryDroplet,
                           "thermo_capillary_droplet",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationThermoCapillaryDroplet,
                           "thermo_capillary_droplet",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::ThermoCapillaryDroplet
