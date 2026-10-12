#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <bitset>
#include <functional>

namespace MeltPoolDG
{
  /**
   * Mark troubled cells, i.e., cells in which the state of at least one subcell fails the given
   * admissibility check.
   *
   * @param matrix_free_context Matrix-free data providing the dof and quadrature indices used to
   * evaluate the solution on the subcells.
   * @param dst Troubled-cell markers with one bitset per cell batch: bit v of dst[cell] is set if
   * lane v of the cell batch is troubled. It is assumed that dst has been resized to the number of
   * cell batches in matrix_free_context.mf.
   * @param solution Solution vector to be checked.
   * @param admissibility_check Function returning for a vectorized subcell state a bitset in which
   * the bits of the admissible lanes are set.
   */
  template <int dim, int n_components, typename number>
  void
  mark_troubled_cells(
    const MatrixFreeContext<dim, number> &matrix_free_context,
    dealii::AlignedVector<std::bitset<dealii::VectorizedArray<number>::size()>> &dst,
    const dealii::LinearAlgebra::distributed::Vector<number>                    &solution,
    const std::function<std::bitset<dealii::VectorizedArray<number>::size()>(
      const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &)>
      &admissibility_check)
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using MarkerType = std::bitset<dealii::VectorizedArray<number>::size()>;

    std::function<void(const dealii::MatrixFree<dim, number> &,
                       dealii::AlignedVector<MarkerType> &,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &)>
      cell_loop = [&](const dealii::MatrixFree<dim, number>       &matrix_free,
                      dealii::AlignedVector<MarkerType>           &marked_cells,
                      const VectorType                            &current_solution,
                      const std::pair<unsigned int, unsigned int> &cell_range) {
        FESubcellEvaluation<dim, n_components, number> fe_subcell_evaluator(
          matrix_free, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            fe_subcell_evaluator.reinit(cell);
            fe_subcell_evaluator.gather_evaluate(current_solution, dealii::EvaluationFlags::values);

            MarkerType cells_marked;

            for (const unsigned int subcell : fe_subcell_evaluator.subcell_indices())
              {
                const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>
                  subcell_state = fe_subcell_evaluator.get_value(subcell);
                cells_marked |= ~admissibility_check(subcell_state);
              }
            fe_subcell_evaluator.set_cell_data(marked_cells, cells_marked);
          }
      };

    matrix_free_context.mf.loop_cell_centric(cell_loop, dst, solution);
  }
} // namespace MeltPoolDG
