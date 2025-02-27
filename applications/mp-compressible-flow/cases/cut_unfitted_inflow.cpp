#include "cut_unfitted_inflow.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationCutUnfittedInflow,
                           "cut_unfitted_inflow",
                           2);
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationCutUnfittedInflow,
                           "cut_unfitted_inflow",
                           3);
} // namespace MeltPoolDG::Simulation::CompressibleFlow
