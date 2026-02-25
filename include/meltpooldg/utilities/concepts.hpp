#pragma once

#include <concepts>

namespace MeltPoolDG
{
  /**
   * A concept defining the requirements for a type to be considered an arithmetic type. This
   * concept is used to ensure that types used in mathematical operations support basic arithmetic
   * operations such as addition, subtraction, multiplication, and division.
   */
  template <typename T>
  concept ArithmeticType = requires(T a, T b) {
    {
      a + b
    } -> std::convertible_to<T>;
    {
      a - b
    } -> std::convertible_to<T>;
    {
      a *b
    } -> std::convertible_to<T>;
    {
      a / b
    } -> std::convertible_to<T>;
  };
} // namespace MeltPoolDG