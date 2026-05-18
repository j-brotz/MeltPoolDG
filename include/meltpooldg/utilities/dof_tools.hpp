#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/full_matrix.h>

#include <meltpooldg/utilities/fe_integrator.hpp>


namespace MeltPoolDG::DoFTools
{
  /**
   * This function creates a n x m interpolation matrix P, to interpolate DoF values
   * per cell from one space (n) given by @param dof_handler_1 to another space (m)
   * given py @param dof_handler_2.
   *
   * The interpolation of cell-wise DoF values x, using the matrix P can be done
   * as follows
   *   _
   *   x   = P   x
   *    i     ij  j
   *
   * with i=0...n-1 and j=0...m-1.
   *
   * @note The row/column indices are sorted in lexicographic order.
   *
   * @note Enable do_sort_lexicographically if the interpolation matrix
   *   should be used in matrix-free loops.
   *
   *
   * ---------------------------------------------------------------------------------
   * Copied from adaflo:
   *
   * https://github.com/kronbichler/adaflo/blob/f873472c43798304bbdb7f0cbeb556061c489020/source/level_set_base.cc#L68-L137
   *
   * @note semantics slightly modified
   * ---------------------------------------------------------------------------------
   */
  template <int dim, typename number>
  dealii::FullMatrix<number>
  create_dof_interpolation_matrix(const dealii::DoFHandler<dim> &dof_handler_1, // e.g. pressure
                                  const dealii::DoFHandler<dim> &dof_handler_2, // e.g. level set
                                  const bool                     do_sort_lexicographically);

  /**
   * Compute from @param values of a given field, @param interpolated_values by
   * means of a given @param interpolation_matrix. Finally, from the interpolated_values
   * the gradients are evaluated.
   *
   * @note The interpolation_matrix should be computed using create_dof_interpolation_matrix().
   *
   */
  template <int dim, typename number>
  void
  compute_gradient_at_interpolated_dof_values(
    FECellIntegrator<dim, 1, number> &values,
    FECellIntegrator<dim, 1, number> &interpolated_values,
    const dealii::FullMatrix<number> &interpolation_matrix);
} // namespace MeltPoolDG::DoFTools
