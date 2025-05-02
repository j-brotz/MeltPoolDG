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

#include <cmath>
#include <cstddef>


namespace MeltPoolDG::VectorTools
{
  template <typename number>
  dealii::VectorizedArray<number>
  compute_mask_narrow_band(const dealii::VectorizedArray<number> &val,
                           const number                           narrow_band_threshold);

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


  template <int dim, typename number>
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
  normalize(const dealii::VectorizedArray<number> &in, const number zero = 1e-16);

  template <int dim, typename number>
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
  normalize(const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &in,
            const number                                                   zero = 1e-16);

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

  /**
   * Calculate the tanh for a vectorized arry @p arg.
   */
  template <int dim, typename number>
  inline dealii::VectorizedArray<number>
  tanh(dealii::VectorizedArray<number> const &arg)
  {
    return (std::exp(arg) - std::exp(-arg)) / (std::exp(arg) + std::exp(-arg));
  }

  /**
   * Compute the hyperbolic tangent of a vectorized data field. The result is returned
   * as vectorized array in the form <tt>{tanh(x[0]), tanh(x[1]), ...,
   * tanh(x[size()-1])}</tt>.
   */
  template <typename number, std::size_t width>
  dealii::VectorizedArray<number>
  tanh(const ::dealii::VectorizedArray<number, width> &x);

  // TODO: Check performance when the functions below are inlined.
  /**
   * Compute the dyadic product of two rank-1 tensors passing a pointer to the start of the values
   * of the two tensors.
   */
  template <int T1_dim, int T2_dim, typename number>
  dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>>
  dyadic_product(const number *a_start, const number *b_start);

  /**
   * Compute the dyadic product of two rank-1 tensors
   */
  template <int T1_dim, int T2_dim, typename number>
  dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>>
  dyadic_product(const dealii::Tensor<1, T1_dim, number> &a,
                 const dealii::Tensor<1, T2_dim, number> &b);

  /**
   * Return the transpose of a dealii::Tensor<dealii::Tensor>
   */
  template <int T1_dim, int T2_dim, typename number>
  dealii::Tensor<1, T2_dim, dealii::Tensor<1, T1_dim, number>>
  transpose(const dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>> &in);

  /**
   * Return the trace of a dealii::Tensor<dealii::Tensor>
   */
  template <int dim, typename number>
  number
  trace(const dealii::Tensor<1, dim, dealii::Tensor<1, dim, number>> &in);

  template <int dim, typename number>
  number
  trace(const dealii::Tensor<2, dim, number> &in);

  /**
   * Return identity matrix
   */
  template <int dim, typename number>
  dealii::Tensor<1, dim, dealii::Tensor<1, dim, number>>
  identity();

  /**
   * Helper functions for matrix-vector and matrix-matrix computations when both matrix and vector
   * are implemented as dealii::Tensor.
   */
  template <int n_rows, int n_columns, typename number>
  dealii::Tensor<1, n_rows, number>
  matrix_vector_product(
    const dealii::Tensor<1, n_rows, dealii::Tensor<1, n_columns, number>> &matrix,
    const dealii::Tensor<1, n_columns, number>                            &vector);

  template <int a, int b, int c, typename number>
  dealii::Tensor<1, a, dealii::Tensor<1, c, number>>
  matrix_matrix_product(const dealii::Tensor<1, a, dealii::Tensor<1, b, number>> &matrix1,
                        const dealii::Tensor<1, b, dealii::Tensor<1, c, number>> &matrix2);
} // namespace MeltPoolDG::VectorTools
