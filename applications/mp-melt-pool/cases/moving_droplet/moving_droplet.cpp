#include "moving_droplet.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::MovingDroplet
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationMovingDroplet, "moving_droplet", 1, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationMovingDroplet, "moving_droplet", 2, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationMovingDroplet, "moving_droplet", 3, double);
} // namespace MeltPoolDG::Simulation::MovingDroplet
