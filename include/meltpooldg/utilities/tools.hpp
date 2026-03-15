#pragma once

#include <tuple>
#include <utility>

namespace MeltPoolDG
{
  /**
   * This function performs an element-wise addition of two tuples, where the first tuple contains
   * references to the elements to be updated, and the second tuple contains the values to be added.
   * The addition is performed in-place on the first tuple. Its functionality is similar to
   * `std::tie` but instead of assigning values, it adds them to the existing values in the first
   * tuple.
   *
   * @tparam Ts Variadic template parameters representing the types of the elements in the tuples.
   *
   * @param lhs A tuple of references to the elements that will be updated with the addition.
   * @param rhs A tuple of values that will be added to the corresponding elements in `lhs`.
   */
  template <typename... Ts>
  void
  tie_add(std::tuple<Ts &...> lhs, const std::tuple<Ts...> &rhs)
  {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(lhs) += std::get<I>(rhs)), ...);
    }(std::index_sequence_for<Ts...>{});
  }
} // namespace MeltPoolDG