/* ---------------------------------------------------------------------
 * Johannes Resch, TUM, July 2024
 *
 * ---------------------------------------------------------------------*/

#pragma once

#include <cmath>
namespace MeltPoolDG::LevelSet
{
  /**
   * The given distance value is transformed to a smooth heaviside function \f$H_\epsilon\f$,
   * which has the property of \f$\int \nabla H_\epsilon=1\f$. This function has its transition
   * region between -2 and 2.
   */
  inline double
  smooth_heaviside_from_distance_value(const double x /*distance*/)
  {
    if (x > 0)
      return 1. - smooth_heaviside_from_distance_value(-x);
    else if (x < -2.)
      return 0;
    else if (x < -1.)
      {
        const double x2 = x * x;
        return (0.125 * (5. * x + x2) +
                0.03125 * (-3. - 2. * x) * std::sqrt(-7. - 12. * x - 4. * x2) -
                0.0625 * std::asin(std::sqrt(2.) * (x + 1.5)) + 23. * 0.03125 -
                dealii::numbers::PI / 64.);
      }
    else
      {
        const double x2 = x * x;
        return (0.125 * (3. * x + x2) -
                0.03125 * (-1. - 2. * x) * std::sqrt(1. - 4. * x - 4. * x2) +
                0.0625 * std::asin(std::sqrt(2.) * (x + 0.5)) + 15. * 0.03125 -
                dealii::numbers::PI / 64.);
      }
  }
} // namespace MeltPoolDG::LevelSet