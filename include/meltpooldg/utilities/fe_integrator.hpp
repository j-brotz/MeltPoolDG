#pragma once

#include <deal.II/base/config.h>

#include <deal.II/matrix_free/fe_evaluation.h>

DEAL_II_NAMESPACE_OPEN

template <int dim,
          int n_components,
          typename number,
          typename VectorizedArrayType = dealii::VectorizedArray<number>>
using FECellIntegrator =
  dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>;


template <int dim,
          int n_components,
          typename number,
          typename VectorizedArrayType = dealii::VectorizedArray<number>>
using FEFaceIntegrator =
  dealii::FEFaceEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>;

DEAL_II_NAMESPACE_CLOSE
