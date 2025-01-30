#include <meltpooldg/core/problem_selector.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/heat/heat_transfer_problem.hpp>
#include <meltpooldg/level_set/level_set_problem.hpp>
#include <meltpooldg/melt_pool/melt_pool_problem.hpp>
#include <meltpooldg/reinitialization/reinitialization_problem.hpp>
/* add your problem here*/

namespace MeltPoolDG
{
  template <int dim>
  std::shared_ptr<ProblemBase<dim>>
  ProblemSelector<dim>::get_problem(const std::string &problem_name)
  {
    if (problem_name == "level_set" || problem_name == "level_set_with_evaporation")
      return std::make_shared<LevelSet::LevelSetProblem<dim>>();

    else if (problem_name == "reinitialization")
      return std::make_shared<LevelSet::ReinitializationProblem<dim>>();

    else if (problem_name == "melt_pool")
      return std::make_shared<MeltPool::MeltPoolProblem<dim>>();

    else if (problem_name == "heat_transfer")
      return std::make_shared<Heat::HeatTransferProblem<dim>>();
    /* add your problem here*/

    else
      AssertThrow(false, ExcMessage("The solver for your requested problem type does not exist"));
  }

  template class ProblemSelector<1>;
  template class ProblemSelector<2>;
  template class ProblemSelector<3>;
} // namespace MeltPoolDG
