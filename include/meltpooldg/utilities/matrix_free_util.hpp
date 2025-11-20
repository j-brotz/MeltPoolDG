#pragma once

#include <deal.II/matrix_free/matrix_free.h>

namespace MeltPoolDG
{

  template <int dim, typename Number>
  struct MatrixFreeContext
  {
    const dealii::MatrixFree<dim, Number> &mf;
    unsigned int                           dof_idx;
    unsigned int                           quad_idx;
  };
} // namespace MeltPoolDG