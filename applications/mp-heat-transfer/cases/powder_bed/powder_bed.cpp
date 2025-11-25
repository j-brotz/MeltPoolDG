#include "powder_bed.templates.hpp"
//
#include <meltpooldg/core/case_registration.hpp>

#include "../../heat_transfer_case.hpp"


namespace MeltPoolDG::Simulation::PowderBed
{
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationPowderBed,
                                     "powder_bed",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationPowderBed,
                                     "powder_bed",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationPowderBed,
                                     "powder_bed",
                                     3,
                                     double);
} // namespace MeltPoolDG::Simulation::PowderBed
