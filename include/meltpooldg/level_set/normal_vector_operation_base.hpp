/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    template <int dim>
    class NormalVectorOperationBase
    {
    public:
      virtual void
      solve() = 0;

      virtual void
      reinit() = 0;

      virtual const LinearAlgebra::distributed::BlockVector<double> &
      get_solution_normal_vector() const = 0;

      virtual LinearAlgebra::distributed::BlockVector<double> &
      get_solution_normal_vector() = 0;

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;
    };

  } // namespace LevelSet
} // namespace MeltPoolDG
