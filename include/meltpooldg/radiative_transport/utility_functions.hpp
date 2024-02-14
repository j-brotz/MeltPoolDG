#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>


using namespace dealii;

namespace MeltPoolDG::RadiativeTransport
{
  template <int dim, typename number = double>
  inline VectorizedArray<number>
  compute_mu(const RadiativeTransportData<number>          &rte_data,
             const VectorizedArray<number>                 &H,
             const Tensor<1, dim, VectorizedArray<number>> &grad_H,
             const Tensor<1, dim, number>                  &laser_direction,
             const double                                   avoid_div_zero_constant)
  {
    // 1. material constant mu
    if (rte_data.absorptivity_type == AbsorptivityType::constant)
      return (
        LevelSet::Tools::interpolate(H,
                                     rte_data.absorptivity_constant_data.absorptivity_gas,
                                     rte_data.absorptivity_constant_data.absorptivity_liquid));

    // 2. gradient based mu : max(0, ∇H * laser_dir *1./(1.- H + ϵ))
    else if (rte_data.absorptivity_type == AbsorptivityType::gradient_based)
      {
        VectorizedArray<number> dummy =
          scalar_product(grad_H, laser_direction) * 1. / (1. - H + avoid_div_zero_constant);
        return compare_and_apply_mask<SIMDComparison::less_than>(dummy,
                                                                 0.,
                                                                 /*true*/ 0.,
                                                                 /*false*/ dummy);
      }

    else
      AssertThrow(false, ExcNotImplemented());
    return VectorizedArray<number>(0);
  }


  /**
   * This function returns a mask to identify potentially singular matrix blocks characterized by
   * values equal to zero. Due to the mathematical forumulation of the radiative transfer equation,
   * this may arise if the intensity and the intensity gradient is zero in pure liquid or pure
   * gaseous state.
   *
   * @param I Intensity values.
   * @param grad_I Intensity gradients.
   * @param H Heaviside values.
   * @param pure_liquid_level_set Upper threshold for heaviside values of a pure liquid state.
   */
  template <int dim, typename number = double>
  inline VectorizedArray<number>
  compute_invalid_mask(const VectorizedArray<number>                 &I,
                       const Tensor<1, dim, VectorizedArray<number>> &grad_I,
                       const VectorizedArray<number>                 &H,
                       const double                                   pure_liquid_level_set)
  {
    // only invalid mask on pure liquid
    VectorizedArray<number> liquid_checker =
      compare_and_apply_mask<SIMDComparison::greater_than>(H, pure_liquid_level_set, 1., 0.);
    if (liquid_checker.sum() > (H.size() - 1))
      return VectorizedArray<number>(1.);

    VectorizedArray<number> intensity_grad_magnitude = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
      intensity_grad_magnitude += std::abs(grad_I[d]);

    // else, if cell is in the gas phase H=0, following conditions needs to be met:
    //   cell must experience no intensity
    //   gradient of intensity AND heaviside IN ANY DIRECTION must be zero
    // this makes sure that invalid_mask avoids the intensity column entirely, even on sides
    return compare_and_apply_mask<SIMDComparison::greater_than>(
      compare_and_apply_mask<SIMDComparison::less_than>(I, VectorizedArray<double>(1e-16), 1., 0.) +
        compare_and_apply_mask<SIMDComparison::greater_than>(
          intensity_grad_magnitude, VectorizedArray<double>(1e-16), 1., 0.),
      2. - 1e-16,
      1.,
      0.);
  }
} // namespace MeltPoolDG::RadiativeTransport