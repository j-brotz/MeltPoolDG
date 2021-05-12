#include <meltpooldg/advection_diffusion/advection_diffusion_operator.hpp>

namespace MeltPoolDG::AdvectionDiffusion
{
  template class AdvectionDiffusionOperator<1, double>;
  template class AdvectionDiffusionOperator<2, double>;
  template class AdvectionDiffusionOperator<3, double>;
} // namespace MeltPoolDG::AdvectionDiffusion
