#include "cut_unfitted_inflow.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  MELTPOOLDG_REGISTER_CASE(::MeltPoolDG::CompressibleFlow::CompressibleFlowCase,
                           SimulationCutUnfittedInflow,
                           "cut_unfitted_inflow",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(::MeltPoolDG::CompressibleFlow::CompressibleFlowCase,
                           SimulationCutUnfittedInflow,
                           "cut_unfitted_inflow",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::CompressibleFlow
