#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_common.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>


namespace MeltPoolDG::VectorTools
{
  template <int dim, int spacedim, typename number>
  void
  convert_fe_system_vector_to_block_vector(
    const dealii::LinearAlgebra::distributed::Vector<number> &in,
    const dealii::DoFHandler<dim, spacedim>                  &dof_handler_fe_system,
    dealii::LinearAlgebra::distributed::BlockVector<number>  &out,
    const dealii::DoFHandler<dim, spacedim>                  &dof_handler);

  template <int dim, int n_components, typename number>
  void
  project_function_to_grid_points(dealii::LinearAlgebra::distributed::Vector<number> &out,
                                  const dealii::Function<dim>                        &function,
                                  const dealii::MatrixFree<dim, number>              &matrix_free,
                                  const unsigned int                                  dof_idx  = 0,
                                  const unsigned int                                  quad_idx = 0);

  template <int dim, int spacedim, typename number>
  void
  convert_block_vector_to_fe_system_vector(
    const dealii::LinearAlgebra::distributed::BlockVector<number> &in,
    const dealii::DoFHandler<dim, spacedim>                       &dof_handler,
    dealii::LinearAlgebra::distributed::Vector<number>            &out,
    const dealii::DoFHandler<dim, spacedim>                       &dof_handler_fe_system);

  /**
   * Converts a `dealii::AlignedVector<dealii::VectorizedArray>`—where each entry of a SIMD vector
   * represents a value for a specific cell in the mesh—into a `dealii::Vector<number>`.
   *
   * In the resulting vector, each value corresponds to one (local) active cell on the
   * triangulation. The index of a value in the output vector matches the index of the
   * corresponding active cell.
   *
   * In a nutshell, this function effectively de-vectorizes cell-wise SIMD data into a standard
   * cell-indexed vector representation.
   *
   * @param mf Matrix-free object used to compute the aligned vector.
   * @param cell_aligned_vector Input vector containing SIMD-packed values per cell.
   * @param tria Triangulation associated with the mesh cells.
   * @param dof_idx Dof index relevant to the matrix-free object.
   * @param quad_idx Quadrature index relevant to the matrix-free object.
   */

  template <int dim, typename number, int n_components = dim>
  dealii::Vector<number>
  convert_matrix_free_cell_aligned_vector_to_vector(
    const MatrixFreeContext<dim, number>                         &mf_context,
    const dealii::AlignedVector<dealii::VectorizedArray<number>> &cell_aligned_vector,
    const dealii::Triangulation<dim>                             &tria)
  {
    dealii::Vector<number> vec_out(tria.n_active_cells());
    for (unsigned cell_batch = 0; cell_batch < mf_context.mf.n_cell_batches(); ++cell_batch)
      {
        FECellIntegrator<dim, n_components, number> phi(mf_context.mf,
                                                        mf_context.dof_idx,
                                                        mf_context.quad_idx);
        phi.reinit(cell_batch);
        const dealii::VectorizedArray<number> cell_batch_indicator =
          phi.read_cell_data(cell_aligned_vector);

        for (unsigned lane = 0; lane < mf_context.mf.n_active_entries_per_cell_batch(cell_batch);
             ++lane)
          {
            const auto cell                    = mf_context.mf.get_cell_iterator(cell_batch, lane);
            vec_out(cell->active_cell_index()) = cell_batch_indicator[lane];
          }
      }
    return vec_out;
  }

  template <typename... T>
  void
  update_ghost_values(const T &...args)
  {
    ((args.update_ghost_values()), ...);
  }

  template <typename... T>
  void
  zero_out_ghost_values(const T &...args)
  {
    ((args.zero_out_ghost_values()), ...);
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim, number>
  to_vector(const dealii::Tensor<1, dim, number> &in)
  {
    return in;
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim, number>
  to_vector(const number &in)
  {
    dealii::Tensor<1, dim, number> vec;

    vec[0] = in;

    return vec;
  }

  template <int dim, typename number, int n_components = dim>
  dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>
  evaluate_function_at_vectorized_points(
    const dealii::Function<dim>                               &func,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &points);

  template <int dim, typename number>
  dealii::VectorizedArray<number>
  evaluate_function_at_vectorized_points(
    const dealii::Function<dim>                               &function,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &p_vectorized,
    const unsigned int                                         component);

  template <int dim, typename number, typename VectorType>
  number
  compute_norm(const VectorType                   &solution,
               const dealii::Triangulation<dim>   &triangulation,
               const dealii::Mapping<dim>         &mapping,
               const dealii::DoFHandler<dim>      &dof_handler,
               const dealii::Quadrature<dim>      &quadrature,
               const dealii::VectorTools::NormType norm_type);

  template <int dim, typename number, typename VectorType>
  number
  compute_norm(const VectorType                    &solution,
               const ScratchData<dim, dim, number> &scratch_data,
               const unsigned int                   dof_idx,
               const unsigned int                   quad_idx,
               const dealii::VectorTools::NormType  norm_type = dealii::VectorTools::L2_norm);

  template <int n_components, int dim, typename VectorType>
  void
  project_vector(const dealii::Mapping<dim>                                       &mapping,
                 const dealii::DoFHandler<dim>                                    &dof,
                 const dealii::AffineConstraints<typename VectorType::value_type> &constraints,
                 const dealii::Quadrature<dim>                                    &quadrature,
                 const VectorType                                                 &vec_in,
                 VectorType                                                       &vec_out);

  /**
   * For a given @p matrix_free object, execute scalar- or vector-valued @p cell_operation
   * on each quadrature point  defined by @p quad_idx and fill them into a
   * DoF-vector @p vec defined by @p dof_idx.
   */
  template <int dim, int n_components, typename T, typename VectorType, typename number>
  void
  fill_dof_vector_from_cell_operation(
    VectorType                                                             &vec,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    unsigned int                                                            dof_idx,
    unsigned int                                                            quad_idx,
    const T                                                                &cell_operation);

  /**
   * Calculate the overall maximum element of a distributed vector @p vec.
   */
  template <typename number>
  number
  max_element(const dealii::LinearAlgebra::distributed::Vector<number> &vec,
              const MPI_Comm                                           &mpi_comm);

  /**
   * Calculate the overall minimum element of a distributed vector @p vec.
   */
  template <typename number>
  number
  min_element(const dealii::LinearAlgebra::distributed::Vector<number> &vec,
              const MPI_Comm                                           &mpi_comm);

} // namespace MeltPoolDG::VectorTools
