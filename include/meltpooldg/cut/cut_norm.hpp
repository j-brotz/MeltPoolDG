#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>

#include <deal.II/numerics/vector_tools_common.h>

#include <memory>
#include <vector>

namespace MeltPoolDG::CutUtil
{
  using NormType = dealii::VectorTools::NormType;

  /**
   * @brief Compute the norm of a solution vector with a non-fitted (cut) domain representation.
   *
   * This function computes either the L1 or L2 norm of the given solution vector on a mesh that may
   * include non-fitted (sub)domain boundaries (e.g., liquid, gas, or intersected cells).
   * The norm is computed with optional two-phase support.
   *
   * @param solution The current solution vector.
   * @param matrix_free The MatrixFree object.
   * @param mapping_info_cells Vector of MappingInfo objects for subdomains (e.g., liquid/gas),
   *        required for intersected cells.
   * @param is_two_phase If true, handles the two-phase setup, processing both components
   *        separately (liquid and gas).
   * @param reference_element Reference finite element used for FEPointEvaluation.
   * @param dof_idx Index of the DoFHandler inside the MatrixFree object.
   * @param quad_idx Index of the quadrature rule inside the MatrixFree object.
   * @param norm_type The type of norm to compute (L1 or L2), defined by NormType enum.
   *
   * @return The computed norm (L1 or L2) over the mesh, correctly reduced across MPI processes.
   *
   * @note Currently implemented only for scalar fields (`n_components == 1`). An assertion fails
   *       if this is not satisfied.
   */
  template <int dim, typename number>
  number
  compute_cut_norm(
    const dealii::LinearAlgebra::distributed::Vector<number>               &solution,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
                            &mapping_info_cells,
    const bool               is_two_phase,
    const dealii::FE_Q<dim> &reference_element,
    const unsigned int       dof_idx,
    const unsigned int       quad_idx,
    const NormType           norm_type = NormType::L2_norm);
} // namespace MeltPoolDG::CutUtil
