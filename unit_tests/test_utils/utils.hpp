#pragma once

#include <gtest/gtest.h>

#include <deal.II/base/config.h>

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

} // namespace MeltPoolDG::TestUtils