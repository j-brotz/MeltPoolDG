#pragma once
#include <meltpooldg/interface/problem_base.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  class ProblemSelector
  {
  public:
    static std::shared_ptr<ProblemBase<dim>>
    get_problem(const ProblemType &problem_name);
  };
} // namespace MeltPoolDG
