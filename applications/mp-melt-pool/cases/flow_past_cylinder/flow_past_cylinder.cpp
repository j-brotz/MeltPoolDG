#include "flow_past_cylinder.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::FlowPastCylinder
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationFlowPastCylinder,
                           "flow_past_cylinder",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationFlowPastCylinder,
                           "flow_past_cylinder",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationFlowPastCylinder,
                           "flow_past_cylinder",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::FlowPastCylinder
