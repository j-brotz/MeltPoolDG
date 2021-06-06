#include <meltpooldg/advection_diffusion/advection_diffusion_problem.hpp>
#include <meltpooldg/flow/two_phase_flow_problem.hpp>
#include <meltpooldg/heat/heat_transfer_problem.hpp>
#include <meltpooldg/interface/problem_selector.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_problem.hpp>
#include <meltpooldg/reinitialization/reinitialization_problem.hpp>
/* add your problem here*/

namespace MeltPoolDG
{
  template <int dim>
  std::shared_ptr<ProblemBase<dim>>
  ProblemSelector<dim>::get_problem(std::string problem_name)
  {
    if (problem_name == "level_set" || problem_name == "level_set_with_evaporation")
      return std::make_shared<LevelSet::LevelSetProblem<dim>>();

    else if (problem_name == "reinitialization")
      return std::make_shared<Reinitialization::ReinitializationProblem<dim>>();

    else if (problem_name == "advection_diffusion")
      return std::make_shared<AdvectionDiffusion::AdvectionDiffusionProblem<dim>>();

    else if (problem_name == "two_phase_flow" || problem_name == "melt_pool" ||
             problem_name == "two_phase_flow_with_evaporation" ||
             problem_name == "melt_pool_with_evaporation" ||
             problem_name == "two_phase_flow_with_heat_transfer")
      return std::make_shared<Flow::TwoPhaseFlowProblem<dim>>();

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
