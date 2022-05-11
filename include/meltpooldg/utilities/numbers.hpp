/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, November 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <cmath>
#include <limits>

namespace dealii::numbers
{
  constexpr double invalid_double = std::numeric_limits<double>::min();

  constexpr auto
  is_invalid(double number)
  {
    return number == invalid_double;
  }

} // namespace dealii::numbers
