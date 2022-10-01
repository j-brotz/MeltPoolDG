/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG
{
  namespace NormalVector
  {
    using namespace dealii;

    template <int dim>
    class NormalVectorOperationBase
    {
    public:
      virtual void
      solve(const LinearAlgebra::distributed::Vector<double> &advected_field) = 0;

      virtual void
      reinit() = 0;

      virtual const LinearAlgebra::distributed::BlockVector<double> &
      get_solution_normal_vector() const = 0;

      virtual LinearAlgebra::distributed::BlockVector<double> &
      get_solution_normal_vector() = 0;
    };

  } // namespace NormalVector
} // namespace MeltPoolDG
