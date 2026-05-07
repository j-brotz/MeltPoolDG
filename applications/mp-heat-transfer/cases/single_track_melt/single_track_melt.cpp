#include "../../../mp-melt-pool/cases/single_track_melt/single_track_melt.templates.hpp"
//
#include <meltpooldg/core/case_registration.hpp>

#include "../../heat_transfer_case.hpp"


namespace MeltPoolDG::Simulation::SingleTrackMelt
{
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationSingleTrackMelt,
                                     "single_track_melt",
                                     3,
                                     double);
} // namespace MeltPoolDG::Simulation::SingleTrackMelt
