#pragma once

#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

namespace MeltPoolDG
{
  BETTER_ENUM(FlowSolverType, char, compressible, incompressible);

  namespace Flow
  {
    // Only linear external force supported
    template <int dim, typename number, int n_components>
    class IncompressibleExternalFluidForce
    {
    public:
      virtual void
      cell_operation(const MatrixFreeContext<dim, number> &matrix_free,
                     unsigned int                          cell_batch_id) = 0;

      virtual dealii::VectorizedArray<number>
      get_damping_coeff_at_q(
        number                                                     time_step_size,
        const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point) = 0;

      virtual dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>
      get_rhs_at_q(number                                                     time_step_size,
                   const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point) = 0;
    };
  } // namespace Flow
} // namespace MeltPoolDG