#include "cut_flow_over_cylinder.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  // Explicit instantiations for the required dimensions
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationCutFlowOverCylinder,
                           "cut_flow_over_cylinder",
                           2);
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationCutFlowOverCylinder,
                           "cut_flow_over_cylinder",
                           3);
} // namespace MeltPoolDG::Simulation::CompressibleFlow
