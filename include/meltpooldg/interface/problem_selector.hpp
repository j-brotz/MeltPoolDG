#pragma once
// MeltPoolDG
#include <meltpooldg/interface/problem_base.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  class ProblemSelector
  {
  public:
    static std::shared_ptr<ProblemBase<dim>>
    get_problem(std::string problem_name);
  };
} // namespace MeltPoolDG
