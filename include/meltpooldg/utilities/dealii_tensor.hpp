#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <cstddef>

namespace dealii
{
  template <typename number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>>
  operator+=(const dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>> &vec,
             const dealii::VectorizedArray<number, N>                       &scalar)
  {
    auto temp = vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>>
  operator+=(const dealii::VectorizedArray<number, N>                       &scalar,
             const dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>> &vec)
  {
    auto temp = vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>>
  operator-=(const dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>> &vec,
             const dealii::VectorizedArray<number, N>                       &scalar)
  {
    auto temp = vec;
    temp[0] -= scalar;

    return temp;
  }

  template <typename number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>>
  operator-=(const dealii::VectorizedArray<number, N>                       &scalar,
             const dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>> &vec)
  {
    auto temp = -vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename number, std::size_t N>
  dealii::VectorizedArray<number, N>
  scalar_product(const dealii::VectorizedArray<number, N>                       &scalar,
                 const dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>> &vec)
  {
    return vec[0] * scalar;
  }

  template <typename number, std::size_t N>
  dealii::VectorizedArray<number, N>
  scalar_product(const dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>> &vec,
                 const dealii::VectorizedArray<number, N>                       &scalar)
  {
    return vec[0] * scalar;
  }
} // namespace dealii