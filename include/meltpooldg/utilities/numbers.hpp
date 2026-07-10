#pragma once

#include <cmath>
#include <numbers>

namespace MeltPoolDG::numbers
{
  // @note We did not use std::numeric_limits<double>::lowest() on purpose.
  // @p invalid_double should serve as a default value for an optional parameter.
  // During parsing, the ParameterHandler converts the default value (e.g.
  // set to invalid_double) to a string via Patterns::Tools::to_string<double>()
  // and then back to a double value. This entails a loss in the number of decimal
  // places and thus round-off errors.
  constexpr double invalid_double = -1e100;

  // Check if a @p number is invalid, if @p number is smaller or equal to
  // numbers::invalid_double.
  inline bool
  is_invalid(const double &number)
  {
    return number <= invalid_double;
  }

  /**
   * @brief Convert an angle from degrees to radians.
   *
   * This function takes an angle specified in degrees and converts it
   * to radians using the relation \f$ \theta_{\mathrm{rad}} = \theta_{\mathrm{deg}} \cdot \pi / 180
   * \f$.
   *
   * @tparam number Numeric type (e.g., float, double).
   * @param angle_deg Angle in degrees.
   * @return Angle in radians.
   */
  template <typename number>
  number
  compute_angle_in_radians(const number angle_deg)
  {
    return angle_deg * number(std::numbers::pi_v<number> / 180.0);
  }

  /// Square root of pi
  constexpr double sqrt_pi = std::numbers::pi_v<double> * std::numbers::inv_sqrtpi_v<double>;
} // namespace MeltPoolDG::numbers
