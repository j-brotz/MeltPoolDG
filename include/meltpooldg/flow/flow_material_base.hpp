/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, TUM, May 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim, typename number = double>
  class MaterialBase
  {
  public:
    virtual void
    reinit(const Tensor<2, dim, VectorizedArray<number>> &velocity_gradient,
           const unsigned int                             cell_idx,
           const unsigned int                             quad_idx) = 0;

    virtual Tensor<2, dim, VectorizedArray<number>>
    get_tau() = 0;

    virtual Tensor<2, dim, VectorizedArray<number>>
    get_d_tau_d_vel_times_vel_correction() = 0;
  };

} // namespace MeltPoolDG::Flow
