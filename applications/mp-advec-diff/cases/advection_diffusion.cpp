#include "advection_diffusion.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::AdvectionDiffusion
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvec,
                           "advection_diffusion",
                           1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvec,
                           "advection_diffusion",
                           2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvec,
                           "advection_diffusion",
                           3);
} // namespace MeltPoolDG::Simulation::AdvectionDiffusion
