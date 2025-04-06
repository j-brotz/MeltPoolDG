#include "advection_diffusion_DG.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::AdvectionDiffusionDG
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecDG,
                           "advection_diffusion_DG",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecDG,
                           "advection_diffusion_DG",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecDG,
                           "advection_diffusion_DG",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::AdvectionDiffusionDG
