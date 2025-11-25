#include "melt_front_propagation.templates.hpp"
//
#include <meltpooldg/core/case_registration.hpp>

#include "../../heat_transfer_case.hpp"


namespace MeltPoolDG::Simulation::MeltFrontPropagation
{

  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     3,
                                     double);

} // namespace MeltPoolDG::Simulation::MeltFrontPropagation
