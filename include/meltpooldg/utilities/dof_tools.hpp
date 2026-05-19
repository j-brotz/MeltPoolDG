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
   * Interpolate DoF values from one finite element space to another.
   *
   * The function evaluates the values stored in @p source at the support points
   * of its finite element space and interpolates them to the support points of
   * the finite element space represented by @p target. The interpolation is
   * performed by applying @p interpolation_matrix to the DoF values of
   * @p source.
   *
   * After the interpolation, the resulting values are submitted as DoF values to
   * @p target using FECellIntegrator::submit_dof_value(). The function does not
   * call FECellIntegrator::evaluate() on @p target afterwards; this must be done
   * by the caller if values or gradients at quadrature points are required.
   *
   * @param source
   *   FECellIntegrator containing the DoF values in the source finite element
   *   space. The function calls FECellIntegrator::evaluate() with
   *   EvaluationFlags::values on this object before accessing its DoF values.
   *
   * @param target
   *   FECellIntegrator into which the interpolated DoF values are submitted.
   *   This object must already be reinitialized for the same cell as @p source.
   *
   * @param interpolation_matrix
   *   Matrix mapping DoF values from the source finite element space to the
   *   target finite element space. Its number of rows must match
   *   @p target.dofs_per_cell and its number of columns must match
   *   @p source.dofs_per_cell. The interpolation_matrix can be computed
   *   using create_dof_interpolation_matrix().
   */
  template <int dim, typename number>
  void
  interpolate_dof_values(FECellIntegrator<dim, 1, number> &source,
                         FECellIntegrator<dim, 1, number> &target,
                         const dealii::FullMatrix<number> &interpolation_matrix);
} // namespace MeltPoolDG::DoFTools
