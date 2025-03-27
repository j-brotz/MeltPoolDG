#pragma once
#include <deal.II/lac/la_parallel_block_vector.h>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
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
    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() const = 0;

    virtual const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() const = 0;

    virtual dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() = 0;
    /*
     *  setter functions
     */
    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() = 0;

    virtual void
    attach_vectors(std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;
  };
} // namespace MeltPoolDG::LevelSet
