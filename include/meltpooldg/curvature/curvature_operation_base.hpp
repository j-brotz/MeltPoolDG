/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/la_parallel_block_vector.h>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    template <int dim>
    class CurvatureOperationBase
    {
    public:
      virtual void
      solve() = 0;

      virtual void
      update_normal_vector() = 0;

      virtual void
      reinit() = 0;

      /*
       *  getter functions
       */
      virtual const LinearAlgebra::distributed::Vector<double> &
      get_curvature() const = 0;

      virtual const LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() const = 0;

      virtual LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() = 0;
      /*
       *  setter functions
       */
      virtual LinearAlgebra::distributed::Vector<double> &
      get_curvature() = 0;

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;
    };

  } // namespace LevelSet
} // namespace MeltPoolDG
