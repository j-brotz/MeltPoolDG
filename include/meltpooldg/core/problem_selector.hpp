#pragma once
#include <meltpooldg/core/problem_base.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  class ProblemSelector
  {
  public:
    static std::shared_ptr<ProblemBase<dim>>
    get_problem(const std::string &problem_name);
  };
} // namespace MeltPoolDG
