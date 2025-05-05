#include "melt_front_propagation.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::MeltFrontPropagation
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationMeltFrontPropagation,
                           "melt_front_propagation",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationMeltFrontPropagation,
                           "melt_front_propagation",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationMeltFrontPropagation,
                           "melt_front_propagation",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::MeltFrontPropagation
