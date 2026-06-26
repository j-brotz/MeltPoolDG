#pragma once

#include <gtest/gtest.h>

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

namespace MeltPoolDG::TestUtils
{
  inline DEAL_II_ALWAYS_INLINE //
    void
    expect_double_eq(const double &a, const double &b)
  {
    EXPECT_DOUBLE_EQ(a, b);
  }

  template <std::size_t size>
  inline DEAL_II_ALWAYS_INLINE //
    void
    expect_double_eq(const dealii::VectorizedArray<double, size> &a,
                     const dealii::VectorizedArray<double, size> &b)
  {
    for (unsigned int i = 0; i < size; ++i)
      expect_double_eq(a[i], b[i]);
  }

  template <int components, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    void
    expect_double_eq(const dealii::Tensor<1, components, number> &a,
                     const dealii::Tensor<1, components, number> &b)
  {
    for (unsigned int i = 0; i < components; ++i)
      expect_double_eq(a[i], b[i]);
  }

  inline DEAL_II_ALWAYS_INLINE //
    void
    expect_near(const double &a, const double &b, const double abs_tol = 1e-12)
  {
    EXPECT_NEAR(a, b, abs_tol);
  }

  template <std::size_t size>
  inline DEAL_II_ALWAYS_INLINE //
    void
    expect_near(const dealii::VectorizedArray<double, size> &a,
                const dealii::VectorizedArray<double, size> &b,
                const double                                 abs_tol = 1e-12)
  {
    for (unsigned int i = 0; i < size; ++i)
      expect_near(a[i], b[i], abs_tol);
  }

  template <int components, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    void
    expect_near(const dealii::Tensor<1, components, number> &a,
                const dealii::Tensor<1, components, number> &b,
                const double                                 abs_tol = 1e-12)
  {
    for (unsigned int i = 0; i < components; ++i)
      expect_near(a[i], b[i], abs_tol);
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    void
    skip_if_not_correct_mpi_process_count(const unsigned int expected_count)
  {
    if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) != expected_count)
      {
        GTEST_SKIP() << "This test requires exactly " << expected_count << " MPI processes.";
      }
  }

} // namespace MeltPoolDG::TestUtils
