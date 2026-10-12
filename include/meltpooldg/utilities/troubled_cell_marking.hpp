#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
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
   * Convert a vectorized lane mask to a bitset. Bit v is set if lane v of @p mask is nonzero.
   *
   * @param mask Vectorized lane mask, e.g., the troubled cell markers of a batch obtained with
   * read_cell_data().
   * @param n_active_lanes Number of active lanes of the cell or face batch. The bits of the
   * remaining lanes are not set.
   */
  template <typename number>
  inline std::bitset<dealii::VectorizedArray<number>::size()>
  vectorized_mask_to_bitset(const dealii::VectorizedArray<number> &mask,
                            const unsigned int                     n_active_lanes)
  {
    AssertIndexRange(n_active_lanes, dealii::VectorizedArray<number>::size() + 1);

    std::bitset<dealii::VectorizedArray<number>::size()> bitset;
    for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
      bitset.set(lane, mask[lane] != number(0.));
    return bitset;
  }

  /**
   * Convert a bitset to a vectorized lane mask, which is 1 in the lanes whose bit is set and 0 in
   * all other lanes.
   *
   * @param bitset The bitset to be converted.
   */
  template <typename number>
  inline dealii::VectorizedArray<number>
  bitset_to_vectorized_mask(const std::bitset<dealii::VectorizedArray<number>::size()> &bitset)
  {
    dealii::VectorizedArray<number> mask = 0.;
    for (unsigned int lane = 0; lane < dealii::VectorizedArray<number>::size(); ++lane)
      if (bitset[lane])
        mask[lane] = 1.;
    return mask;
  }

  /**
   * Mark troubled cells, i.e., cells in which the state of at least one subcell fails the given
   * admissibility check.
   *
   * @param matrix_free_context Matrix-free data providing the dof and quadrature indices used to
   * evaluate the solution on the subcells.
   * @param dst Troubled cell markers with one vectorized array per cell batch: lane v of dst[cell]
   * is 1 if lane v of the cell batch is troubled and 0 otherwise. Use vectorized_mask_to_bitset()
   * to convert the markers of a batch to a bitset. It is assumed that dst has been initialized with
   * dealii::MatrixFree::initialize_cell_data_vector().
   * @param solution Solution vector to be checked.
   * @param admissibility_check Function returning for a vectorized subcell state a bitset in which
   * the bits of the admissible lanes are set.
   */
  template <int dim, int n_components, typename number>
  void
  mark_troubled_cells(const MatrixFreeContext<dim, number>                     &matrix_free_context,
                      dealii::AlignedVector<dealii::VectorizedArray<number>>   &dst,
                      const dealii::LinearAlgebra::distributed::Vector<number> &solution,
                      const std::function<std::bitset<dealii::VectorizedArray<number>::size()>(
                        const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &)>
                        &admissibility_check)
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using MarkerType = std::bitset<dealii::VectorizedArray<number>::size()>;

    std::function<void(const dealii::MatrixFree<dim, number> &,
                       dealii::AlignedVector<dealii::VectorizedArray<number>> &,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &)>
      cell_loop = [&](const dealii::MatrixFree<dim, number>                  &matrix_free,
                      dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells,
                      const VectorType                                       &current_solution,
                      const std::pair<unsigned int, unsigned int>            &cell_range) {
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
            fe_subcell_evaluator.set_cell_data(marked_cells,
                                               bitset_to_vectorized_mask<number>(cells_marked));
          }
      };

    matrix_free_context.mf.loop_cell_centric(cell_loop, dst, solution);
  }
} // namespace MeltPoolDG
