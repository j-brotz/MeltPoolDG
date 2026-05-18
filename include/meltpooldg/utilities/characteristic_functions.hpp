#pragma once

#include <deal.II/base/numbers.h>

#include <cmath>


namespace MeltPoolDG::CharacteristicFunctions
{
  template <typename number>
  inline number
  tanh_characteristic_function(const number &distance, const number &eps)
  {
    return std::tanh(distance / (2. * eps));
  }

  template <typename number>
  inline number
  smoothed_heaviside(const number &distance, const number &eps)
  {
    if (distance > eps)
      return 1;
    else if (distance <= -eps)
      return 0;
    else
      return (distance + eps) / (2. * eps) +
             1. / (2. * dealii::numbers::PI) * std::sin(dealii::numbers::PI * distance / eps);
  }

  /**
   * This function returns heaviside values for a given VectorizedArray. The limit to
   * distinguish between 0 and 1 can be adjusted by the argument "limit". This function is
   * particularly suited in the context of MatrixFree routines.
   */
  template <typename number>
  number
  heaviside(const number in, const number limit = 0.0)
  {
    return in > limit ? 1.0 : 0.0;
  }

  template <typename number>
  std::vector<number>
  heaviside(const std::vector<number> in, const number limit = 0.0)
  {
    std::vector<number> out(in.size());
    for (unsigned int i = 0; i < in.size(); ++i)
      out[i] = heaviside(in[i], limit);

    return out;
  }

  template <typename number>
  dealii::VectorizedArray<number>
  heaviside(const dealii::VectorizedArray<number> &in, const number limit = 0.0)
  {
    return compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
      in, dealii::VectorizedArray<number>(limit), 1.0, 0.0);
  }

  template <typename number>
  inline int
  sgn(const number &x)
  {
    return x < 0 ? -1 : 1;
  }
} // namespace MeltPoolDG::CharacteristicFunctions
