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
  heaviside(const number &distance, const number &eps)
  {
    if (distance > eps)
      return 1;
    else if (distance <= -eps)
      return 0;
    else
      return (distance + eps) / (2. * eps) +
             1. / (2. * dealii::numbers::PI) * std::sin(dealii::numbers::PI * distance / eps);
  }

  template <typename number>
  inline int
  sgn(const number &x)
  {
    return x < 0 ? -1 : 1;
  }

  template <typename number>
  inline number
  normalize(const number &x, const number &x_min, const number &x_max)
  {
    return (x - x_min) / (x_max - x_min);
  }
} // namespace MeltPoolDG::CharacteristicFunctions