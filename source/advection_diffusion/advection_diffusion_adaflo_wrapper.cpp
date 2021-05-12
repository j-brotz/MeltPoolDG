#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <meltpooldg/advection_diffusion/advection_diffusion_adaflo_wrapper.hpp>

namespace MeltPoolDG::AdvectionDiffusion
{
  template class AdvectionDiffusionOperationAdaflo<1>;
  template class AdvectionDiffusionOperationAdaflo<2>;
  template class AdvectionDiffusionOperationAdaflo<3>;
} // namespace MeltPoolDG::AdvectionDiffusion

#endif
