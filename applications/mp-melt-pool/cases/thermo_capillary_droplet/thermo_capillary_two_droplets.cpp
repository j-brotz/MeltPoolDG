#include "thermo_capillary_two_droplets.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::ThermoCapillaryTwoDroplets
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationThermoCapillaryTwoDroplets,
                           "thermo_capillary_two_droplets",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationThermoCapillaryTwoDroplets,
                           "thermo_capillary_two_droplets",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationThermoCapillaryTwoDroplets,
                           "thermo_capillary_two_droplets",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::ThermoCapillaryTwoDroplets
