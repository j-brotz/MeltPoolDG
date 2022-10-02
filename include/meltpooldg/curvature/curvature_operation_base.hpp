/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/la_parallel_block_vector.h>

namespace MeltPoolDG
{
  namespace Curvature
  {
    using namespace dealii;

    template <int dim>
    class CurvatureOperationBase
    {
    public:
      virtual void
      solve(const LinearAlgebra::distributed::Vector<double> &advected_field) = 0;

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
    };

  } // namespace Curvature
} // namespace MeltPoolDG
