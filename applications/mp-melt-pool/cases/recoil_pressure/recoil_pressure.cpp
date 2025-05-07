#include "recoil_pressure.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::RecoilPressure
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationRecoilPressure, "recoil_pressure", 1, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationRecoilPressure, "recoil_pressure", 2, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationRecoilPressure, "recoil_pressure", 3, double);
} // namespace MeltPoolDG::Simulation::RecoilPressure
