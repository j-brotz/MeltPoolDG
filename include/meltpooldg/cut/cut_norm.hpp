#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>

#include <memory>
#include <vector>

namespace MeltPoolDG::CutUtil
{
  template <int dim, typename number>
  number
  compute_cut_L2_norm(
    const dealii::LinearAlgebra::distributed::Vector<number>               &solution,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
                            &mapping_info_cells,
    const bool               is_two_phase,
    const dealii::FE_Q<dim> &reference_element,
    const unsigned int       dof_idx,
    const unsigned int       quad_idx);
}