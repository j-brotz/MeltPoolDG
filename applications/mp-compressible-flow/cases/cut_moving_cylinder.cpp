#include "cut_moving_cylinder.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  // Explicit instantiations for the required dimensions and floating point number types
  MELTPOOLDG_REGISTER_CASE(::MeltPoolDG::CompressibleFlow::CompressibleFlowCase,
                           SimulationCutMovingCylinder,
                           "cut_moving_cylinder",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(::MeltPoolDG::CompressibleFlow::CompressibleFlowCase,
                           SimulationCutMovingCylinder,
                           "cut_moving_cylinder",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::CompressibleFlow
