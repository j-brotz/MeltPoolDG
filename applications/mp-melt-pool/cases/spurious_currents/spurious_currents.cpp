#include "spurious_currents.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::SpuriousCurrents
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationSpuriousCurrents,
                           "spurious_currents",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationSpuriousCurrents,
                           "spurious_currents",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationSpuriousCurrents,
                           "spurious_currents",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::SpuriousCurrents
