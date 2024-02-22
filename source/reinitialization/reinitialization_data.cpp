#include <deal.II/base/patterns.h>

#include <meltpooldg/reinitialization/reinitialization_data.hpp>

namespace MeltPoolDG::Reinitialization
{
  template <typename number>
  ReinitializationData<number>::ReinitializationData()
  {
    linear_solver.solver_type         = LinearSolverType::CG;
    linear_solver.preconditioner_type = PreconditionerType::Diagonal;
  }

  template <typename number>
  void
  ReinitializationData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("reinitialization");
    {
      prm.add_parameter("reinit max n steps",
                        max_n_steps,
                        "Sets the maximum number of reinitialization steps");
      prm.add_parameter("reinit constant epsilon",
                        constant_epsilon,
                        "Defines the length parameter of the level set function to be constant and"
                        "not to dependent on the mesh size (default: -1.0 i.e. grid size dependent"
                        "which can be controlled by reinit_epsilon_scale_factor");
      prm.add_parameter(
        "reinit scale factor epsilon",
        scale_factor_epsilon,
        "Defines the scaling factor of the diffusion parameter in the reinitialization "
        "equation; the scaling factor is multipled by the mesh size (default: 0.5 i.e. eps=0.5*h_min");
      prm.add_parameter("reinit modeltype",
                        modeltype,
                        "Sets the type of reinitialization model that should be used.");
      prm.add_parameter(
        "reinit implementation",
        implementation,
        "Choose the corresponding implementation of the reinitialization operation.",
        dealii::Patterns::Selection("meltpooldg|adaflo"));

      predictor.add_parameters(prm);
      linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  ReinitializationData<number>::post()
  {
    predictor.post();
  }

  template struct ReinitializationData<double>;
} // namespace MeltPoolDG::Reinitialization
