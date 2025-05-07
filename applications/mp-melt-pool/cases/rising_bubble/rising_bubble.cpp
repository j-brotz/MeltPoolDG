#include "rising_bubble.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::RisingBubble
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationRisingBubble, "rising_bubble", 1, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationRisingBubble, "rising_bubble", 2, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationRisingBubble, "rising_bubble", 3, double);
} // namespace MeltPoolDG::Simulation::RisingBubble
