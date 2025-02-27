#include "advection_diffusion_DG.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::AdvectionDiffusionDG
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecDG,
                           "advection_diffusion_DG",
                           1);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecDG,
                           "advection_diffusion_DG",
                           2);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecDG,
                           "advection_diffusion_DG",
                           3);
} // namespace MeltPoolDG::Simulation::AdvectionDiffusionDG
