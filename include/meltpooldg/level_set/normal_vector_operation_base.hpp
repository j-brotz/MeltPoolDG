#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class NormalVectorOperationBase
  {
  public:
    virtual void
    solve() = 0;

    virtual void
    reinit() = 0;

    virtual const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_solution_normal_vector() const = 0;

    virtual dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_solution_normal_vector() = 0;

    virtual void
    attach_vectors(std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;
  };

} // namespace MeltPoolDG::LevelSet
