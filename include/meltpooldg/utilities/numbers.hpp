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
  // @note We did not use std::numeric_limits<double>::lowest() on purpose.
  // @p invalid_double should serve as a default value for an optional parameter.
  // During parsing, the ParameterHandler converts the default value (e.g.
  // set to invalid_double) to a string via Patterns::Tools::to_string<double>()
  // and then back to a double value. This entails a loss in the number of decimal
  // places and thus round-off errors.
  const double invalid_double = -1e100;

  // Check if a @p number is invalid, if @p number is smaller or equal to
  // numbers::invalid_double.
  inline bool
  is_invalid(const double &number)
  {
    return number <= invalid_double;
  }

} // namespace dealii::numbers
