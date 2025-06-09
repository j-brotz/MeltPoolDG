/**
 * @brief Type definitions for the compressible flow implementations.
 */

#pragma once

#include <deal.II/base/vectorization.h>

namespace MeltPoolDG::Flow
{
  /**
   * Struct providing type aliases that might be useful in the compressible flow implementations.
   */
  struct CompressibleFlowTypes
  {
    /**
     * Type of the conserved variables []
     */
    template <int dim, typename number>
    using ConservedVariablesType = dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>;

    /**
     * Type of the gradient of the conserved variables []
     */
    template <int dim, typename number>
    using ConservedVariablesGradType =
      dealii::Tensor<1, dim + 2, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;
  };
} // namespace MeltPoolDG::Flow
