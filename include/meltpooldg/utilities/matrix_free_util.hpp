#pragma once

#include <deal.II/matrix_free/matrix_free.h>

namespace MeltPoolDG
{

  template <int dim, typename Number>
  struct MatrixFreeContext
  {
    const dealii::MatrixFree<dim, Number> &mf;
    unsigned int                           dof_idx;
    unsigned int                           quad_idx;
  };

  /**
   * Helper function that returns the values at a given quadrature point (specified by `q_index`)
   * using the provided finite element evaluation object (`fe_eval`) and ensures that the result
   * is always returned as a `dealii::Tensor`.
   *
   * In deal.II, `FEValues` and similar objects return a `VectorizedArray` directly when there is
   * only a single component. However, the algorithms in this codebase are written to operate on
   * tensors for both 2D and 3D problems and for systems with multiple components. This function
   * provides a uniform interface by converting single-component vectorized values into a
   * `Tensor<1,1,VectorizedArray>` while leaving multi-component values as-is.
   *
   * @param fe_eval The finite element evaluation object.
   * @param q_index Index of the quadrature point.

   * @return Value at the specified quadrature point as a `dealii::Tensor`.
   */
  template <typename FeEval>
  dealii::Tensor<1, FeEval::n_components, dealii::VectorizedArray<typename FeEval::number_type>>
  fe_evaluation_tensor_value_at_q(const FeEval &fe_eval, const unsigned q_index)
  {
    using ValueType           = typename FeEval::value_type;
    using VectorizedArrayType = dealii::VectorizedArray<typename FeEval::number_type>;

    if constexpr (std::is_same_v<ValueType, VectorizedArrayType>)
      {
        dealii::Tensor<1, 1, VectorizedArrayType> t;
        t[0] = fe_eval.get_value(q_index);
        return t;
      }
    else
      return fe_eval.get_value(q_index);
  }

  /**
   * This function returns the cell iterators corresponding to the active SIMD
   * lanes of the specified cell batch.
   *
   * @param mf The matrix-free object defining the cell batch of interest.
   * @param cell_batch_id Index of the cell batch.
   */
  template <int dim, typename number>
  std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>>
  cells_in_cell_batch(const dealii::MatrixFree<dim, number> &mf, const unsigned int cell_batch_id)
  {
    unsigned int n_active_lanes = mf.n_active_entries_per_cell_batch(cell_batch_id);

    std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cells;
    cells.reserve(n_active_lanes);

    for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
      cells.push_back(mf.get_cell_iterator(cell_batch_id, lane));

    return cells;
  }
} // namespace MeltPoolDG