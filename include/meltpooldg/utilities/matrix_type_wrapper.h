/**
 * @brief A wrapper for a deal.II like matrix type object. The purpose of this wrapper is to provide
 * an object which has a member function vmult() that can be interpreted as a matrix vector
 * product.
 */

#pragma once

#include <functional>
#include <utility>

namespace MeltPoolDG
{
  template <typename VectorType>
  struct MatrixTypeObject
  {
    /**
     * Constructor, setting the function computing the actual matrix vector product in the member
     * function @f vmult().
     *
     * @param alias_function Fucntion to be called in @f vmult().
     */
    explicit MatrixTypeObject(
      const std::function<void(VectorType &, const VectorType &)> &alias_function)
      : alias_function(std::move(alias_function))
    {}

    /**
     * Apply the function passed to the constructor of this object. The exact functionality of this
     * function therefore depends on the user.
     *
     * @param dst Destination, in which the result is stored.
     * @param src Additional vector used during the calculation.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      alias_function(dst, src);
    };

  private:
    const std::function<void(VectorType &, const VectorType &)> alias_function;
  };
} // namespace MeltPoolDG
