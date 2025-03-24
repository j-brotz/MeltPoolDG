#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>


namespace MeltPoolDG::RadiativeTransport
{
  template <int dim, typename number>
  inline dealii::VectorizedArray<number>
  compute_mu(const RadiativeTransportData<number>                          &rte_data,
             const dealii::VectorizedArray<number>                         &H,
             const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &grad_H,
             const dealii::Tensor<1, dim, number>                          &laser_direction,
             const number                                                   avoid_div_zero_constant)
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
        dealii::VectorizedArray<number> dummy =
          scalar_product(grad_H, laser_direction) * 1. / (1. - H + avoid_div_zero_constant);
        return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(dummy,
                                                                                 0.,
                                                                                 /*true*/ 0.,
                                                                                 /*false*/ dummy);
      }

    else
      AssertThrow(false, dealii::ExcNotImplemented());
    return dealii::VectorizedArray<number>(0);
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
  template <int dim, typename number>
  inline dealii::VectorizedArray<number>
  compute_invalid_mask(const dealii::VectorizedArray<number>                         &I,
                       const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &grad_I,
                       const dealii::VectorizedArray<number>                         &H,
                       const number pure_liquid_level_set)
  {
    // only invalid mask on pure liquid
    dealii::VectorizedArray<number> liquid_checker =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(H,
                                                                           pure_liquid_level_set,
                                                                           1.,
                                                                           0.);
    if (liquid_checker.sum() > H.size() - 1)
      return dealii::VectorizedArray<number>(1.);

    dealii::VectorizedArray<number> intensity_grad_magnitude = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
      intensity_grad_magnitude += std::abs(grad_I[d]);

    // else, if cell is in the gas phase H=0, following conditions needs to be met:
    //   cell must experience no intensity
    //   gradient of intensity AND heaviside IN ANY DIRECTION must be zero
    // this makes sure that invalid_mask avoids the intensity column entirely, even on sides
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
      dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
        I, dealii::VectorizedArray<number>(1e-16), 1., 0.) +
        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
          intensity_grad_magnitude, dealii::VectorizedArray<number>(1e-16), 1., 0.),
      2. - 1e-16,
      1.,
      0.);
  }
} // namespace MeltPoolDG::RadiativeTransport