#include <meltpooldg/advection_diffusion/advection_diffusion_problem.hpp>

namespace MeltPoolDG::AdvectionDiffusion
{
  template class AdvectionDiffusionProblem<1>;
  template class AdvectionDiffusionProblem<2>;
  template class AdvectionDiffusionProblem<3>;
} // namespace MeltPoolDG::AdvectionDiffusion
