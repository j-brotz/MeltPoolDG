#pragma once

#include <deal.II/base/point.h>

#include <vector>


namespace MeltPoolDG::Utility
{
  /**
   * Starting from a @p starting_point, generate points along the given @p
   * unit_vec with a given number of steps @p n_inc_per_side and ranging up to a
   * @p max_distance_per_side. If @p bidirectional is set to true, points on both
   * sides of the @p starting_point are generated.
   */
  template <int dim, typename vector, typename number>
  void
  generate_points_along_vector(std::vector<dealii::Point<dim>> &points_normal_to_interface,
                               const dealii::Point<dim>        &starting_point,
                               const vector                    &unit_vec,
                               const number                     max_distance_per_side,
                               const unsigned int               n_inc_per_side,
                               const bool                       bidirectional = true)
  {
    const number step = max_distance_per_side / n_inc_per_side;
    for (int n = n_inc_per_side; n >= 0; --n)
      points_normal_to_interface.emplace_back(starting_point + unit_vec * n * step);
    if (bidirectional)
      {
        for (unsigned int n = 1; n <= n_inc_per_side; ++n)
          points_normal_to_interface.emplace_back(starting_point - unit_vec * n * step);
      }
  }
} // namespace MeltPoolDG::Utility
