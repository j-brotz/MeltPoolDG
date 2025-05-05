#pragma once

#include <meltpooldg/utilities/vector_tools.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/operators.h>

#include <type_traits>
#include <vector>

namespace MeltPoolDG::VectorTools
{
  template <int dim, typename number>
  dealii::VectorizedArray<number>
  evaluate_function_at_vectorized_points(
    const dealii::Function<dim>                               &function,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &p_vectorized,
    const unsigned int                                         component)
  {
    dealii::VectorizedArray<number> result;
    for (unsigned int v = 0; v < dealii::VectorizedArray<number>::size(); ++v)
      {
        dealii::Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        result[v] = function.value(p, component);
      }
    return result;
  }

  template <int dim, typename number, int n_components>
  dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>
  evaluate_function_at_vectorized_points(
    const dealii::Function<dim>                               &func,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &points)
  {
    AssertDimension(func.n_components, n_components);

    dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> vec;

    for (unsigned int v = 0; v < dealii::VectorizedArray<number>::size(); ++v)
      {
        dealii::Point<dim> point_v;

        for (unsigned int d = 0; d < dim; ++d)
          point_v[d] = points[d][v];

        for (unsigned int d = 0; d < n_components; ++d)
          vec[d][v] = func.value(point_v, d);
      }
    return vec;
  }

  template <int dim, int n_components, typename T, typename VectorType, typename number>
  void
  fill_dof_vector_from_cell_operation(
    VectorType                                                             &vec,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    unsigned int                                                            dof_idx,
    unsigned int                                                            quad_idx,
    const T                                                                &cell_operation)
  {
    FECellIntegrator<dim, n_components, number> fe_eval(matrix_free, dof_idx, quad_idx);

    dealii::MatrixFreeOperators::
      CellwiseInverseMassMatrix<dim, -1, n_components, number, dealii::VectorizedArray<number>>
        inverse_mass_matrix(fe_eval);

    const auto &lexicographic_numbering =
      matrix_free.get_shape_info(dof_idx, quad_idx).lexicographic_numbering;

    VectorType weights;
    weights.reinit(vec);
    std::vector<number> ones(fe_eval.dofs_per_cell, 1.0);

    std::vector<dealii::types::global_dof_index> dof_indices(fe_eval.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> dof_indices_mf(fe_eval.dofs_per_cell);
    std::vector<number>                          dof_values(fe_eval.dofs_per_cell);

    dealii::AffineConstraints<number> dummy;

    vec = 0.0;

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        fe_eval.reinit(cell);

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            const auto temp = cell_operation(cell, q);
            for (int c = 0; c < n_components; ++c)
              if constexpr (std::is_same<typename std::remove_const<decltype(temp)>::type,
                                         dealii::VectorizedArray<number>>::value)
                {
                  static_assert(n_components == 1,
                                "The path should be only accessed for a single component.");
                  fe_eval.begin_values()[q] = temp;
                }
              else if constexpr (
                std::is_same<
                  typename std::remove_const<decltype(temp)>::type,
                  dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>>::value)
                {
                  fe_eval.begin_values()[c * fe_eval.n_q_points + q] = temp[c];
                }
              else
                {
                  Assert(false, dealii::ExcNotImplemented());
                }
          }
        inverse_mass_matrix.transform_from_q_points_to_basis(n_components,
                                                             fe_eval.begin_values(),
                                                             fe_eval.begin_dof_values());

        for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
          {
            matrix_free.get_cell_iterator(cell, v, dof_idx)->get_dof_indices(dof_indices);

            for (unsigned int j = 0; j < dof_indices.size(); ++j)
              {
                dof_indices_mf[j] = dof_indices[lexicographic_numbering[j]];
                dof_values[j]     = fe_eval.begin_dof_values()[j][v];
              }

            dummy.distribute_local_to_global(dof_values, dof_indices_mf, vec);
            dummy.distribute_local_to_global(ones, dof_indices_mf, weights);
          }
      }

    vec.compress(dealii::VectorOperation::add);
    weights.compress(dealii::VectorOperation::add);

    for (unsigned int i = 0; i < vec.locally_owned_size(); ++i)
      if (weights.local_element(i) != 0.0)
        vec.local_element(i) /= weights.local_element(i);
  }
} // namespace MeltPoolDG::VectorTools