#include "am_powder_bed.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CfdDem
{
  // Explicit instantiations for the required dimensions and floating point number type
  MELTPOOLDG_REGISTER_CASE(CfdDemCase, SimulationAMPowderBed, "am_powder_bed", 2, double);
  MELTPOOLDG_REGISTER_CASE(CfdDemCase, SimulationAMPowderBed, "am_powder_bed", 3, double);
} // namespace MeltPoolDG::Simulation::CfdDem
