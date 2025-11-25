#include "../../../mp-heat-transfer/cases/melt_front_propagation/melt_front_propagation.templates.hpp"
//
#include <meltpooldg/core/case_registration.hpp>

#include "../../melt_pool_case.hpp"


namespace MeltPoolDG::Simulation::MeltFrontPropagation
{
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::MeltPoolCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::MeltPoolCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::MeltPoolCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     3,
                                     double);
} // namespace MeltPoolDG::Simulation::MeltFrontPropagation
