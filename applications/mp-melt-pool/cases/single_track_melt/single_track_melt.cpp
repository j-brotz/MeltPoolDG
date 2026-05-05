#include "single_track_melt.templates.hpp"
//
#include <meltpooldg/core/case_registration.hpp>

#include "../../melt_pool_case.hpp"


namespace MeltPoolDG::Simulation::SingleTrackMelt
{
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolCase,
                                     SimulationSingleTrackMelt,
                                     "single_track_melt",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolCase,
                                     SimulationSingleTrackMelt,
                                     "single_track_melt",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolCase,
                                     SimulationSingleTrackMelt,
                                     "single_track_melt",
                                     3,
                                     double);
} // namespace MeltPoolDG::Simulation::SingleTrackMelt
