#pragma once

#include <deal.II/base/types.h>

#include <utility>

namespace MeltPoolDG
{
  /**
   * face numbering according to the colorize flag of dealii::GridGenerator::hyper_rectangle
   *
   * \param lower_bc and \param upper_bc:
   * boundaries perpendicular to the vertical axis i.e. faces in dim-1 direction the vertical axis
   * in the z-axis in 3D and the y-axis in 2D
   *
   * \param left_bc and \param right_bc: boundaries aligned
   * with vertical axis, perpendicular to the x-axis
   *
   * \param front_bc and \param back_bc: boundaries
   * aligned with vertical axis, perpendicular to the y-axis, only in 3D
   */
  template <int dim>
  constexpr std::tuple<dealii::types::boundary_id, /* lower_bc */
                       dealii::types::boundary_id, /* upper_bc */
                       dealii::types::boundary_id, /* left_bc */
                       dealii::types::boundary_id, /* right_bc */
                       dealii::types::boundary_id, /* front_bc */
                       dealii::types::boundary_id  /* back_bc */
                       >
  get_colorized_rectangle_boundary_ids();

  // template specializations

  template <>
  constexpr std::tuple<dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id>
  get_colorized_rectangle_boundary_ids<1>()
  {
    return {0 /* lower_bc */,
            1 /* upper_bc */,
            dealii::numbers::invalid_boundary_id /* left_bc */,
            dealii::numbers::invalid_boundary_id /* right_bc */,
            dealii::numbers::invalid_boundary_id /* front_bc */,
            dealii::numbers::invalid_boundary_id /* back_bc */};
  }

  template <>
  constexpr std::tuple<dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id>
  get_colorized_rectangle_boundary_ids<2>()
  {
    return {2 /* lower_bc */,
            3 /* upper_bc */,
            0 /* left_bc */,
            1 /* right_bc */,
            dealii::numbers::invalid_boundary_id /* front_bc */,
            dealii::numbers::invalid_boundary_id /* back_bc */};
  }

  template <>
  constexpr std::tuple<dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id,
                       dealii::types::boundary_id>
  get_colorized_rectangle_boundary_ids<3>()
  {
    return {4 /* lower_bc */,
            5 /* upper_bc */,
            0 /* left_bc */,
            1 /* right_bc */,
            2 /* front_bc */,
            3 /* back_bc */};
  }
} // namespace MeltPoolDG