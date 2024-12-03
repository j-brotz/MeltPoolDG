/**
 * @brief Implementation of functions defined by C++23.
 */

#pragma once

#include <iterator>

namespace MeltPoolDG::Utils
{
  // Similar to std::ranges::contains. Enforces the container to have a begin() and end() iterator.
  template <typename Container>
  concept range = requires(Container &t) {
    {
      // Return an iterator to the first element of the container
      t.begin()
    } -> std::input_or_output_iterator;
    {
      // Return an iterator to one element behind the last element of the container
      t.end()
    } -> std::input_or_output_iterator;
  };

  template <range Container>
  bool
  contains(const Container &c, typename Container::value_type const value)
  {
    return (std::find(c.begin(), c.end(), value) != c.end());
  }
} // namespace MeltPoolDG::Utils
