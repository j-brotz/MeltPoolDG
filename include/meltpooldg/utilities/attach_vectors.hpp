#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <functional>
#include <utility>
#include <vector>

namespace MeltPoolDG
{
  /**
   * Type alias definitions that are used to collect pointers to DoFHandlers and their respective
   * DoF-Vectors.
   */
  template <int dim, typename VectorType>
  using DoFHandlerAndVectorDataType = std::vector<
    std::pair<const dealii::DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>;

  template <int dim, typename VectorType>
  using AttachDoFHandlerAndVectorsType =
    std::function<void(DoFHandlerAndVectorDataType<dim, VectorType> &)>;

} // namespace MeltPoolDG