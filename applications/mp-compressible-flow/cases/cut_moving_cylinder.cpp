#include "cut_moving_cylinder.hpp"

#include <meltpooldg/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  // Explicit instantiations for the required dimensions
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationCutMovingCylinder,
                           "cut_moving_cylinder",
                           2);
  MELTPOOLDG_REGISTER_CASE(Flow::CompressibleFlowCase,
                           SimulationCutMovingCylinder,
                           "cut_moving_cylinder",
                           3);
} // namespace MeltPoolDG::Simulation::CompressibleFlow
