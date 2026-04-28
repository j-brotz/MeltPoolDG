#include "hyper_rectangle_box_case.hpp"

#include <meltpooldg/core/case_registration.hpp>

#include "../meltpool_vapor_flow_case.hpp"

namespace MeltPoolDG::Simulation::MeltPoolVaporFlow
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolDG::MeltPoolVaporFlow::Case,
                           SimulationHyperRectangleBox,
                           "hyper_rectangle_box",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolDG::MeltPoolVaporFlow::Case,
                           SimulationHyperRectangleBox,
                           "hyper_rectangle_box",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::MeltPoolVaporFlow
