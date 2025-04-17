#pragma once
#include <meltpooldg/core/problem_base.hpp>

namespace MeltPoolDG
{
  template <int dim, typename number>
  class ProblemSelector
  {
  public:
    static std::shared_ptr<ProblemBase<dim, number>>
    get_problem(const std::string &problem_name, std::unique_ptr<MeltPoolCase<dim, number>> sim);
  };
} // namespace MeltPoolDG
