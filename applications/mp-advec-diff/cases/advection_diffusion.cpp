#include "advection_diffusion.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::AdvectionDiffusion
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvec,
                           "advection_diffusion",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvec,
                           "advection_diffusion",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvec,
                           "advection_diffusion",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::AdvectionDiffusion
