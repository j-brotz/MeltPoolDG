#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/reinitialization/reinitialization_data.hpp>

namespace MeltPoolDG::LevelSet
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
      prm.add_parameter("enable", enable, "Set to true to activate reinitialization.");
      prm.add_parameter(
        "n initial steps",
        n_initial_steps,
        "Defines the number of initial reinitialization steps of the level set function. "
        "In the default case, the number is set equal to the number of max n steps.");
      prm.add_parameter("max n steps",
                        max_n_steps,
                        "Sets the maximum number of reinitialization steps");
      prm.add_parameter("tolerance",
                        tolerance,
                        "Set the tolerance for reinitialization. If the "
                        "maximum change of the level set field, i.e. ||ΔФ||∞, exceeds the "
                        "tolerance, reinitialization steps will be performed.");
      prm.add_parameter("type",
                        modeltype,
                        "Sets the type of reinitialization model that should be used.");
      prm.add_parameter(
        "implementation",
        implementation,
        "Choose the corresponding implementation of the reinitialization operation.",
        dealii::Patterns::Selection("meltpooldg|adaflo"));

      prm.enter_subsection("interface thickness parameter");
      {
        prm.add_parameter("type",
                          interface_thickness_parameter.type,
                          "Choose the value type of the interface thickness parameter.");
        prm.add_parameter("val",
                          interface_thickness_parameter.value,
                          "Defines the value of the chosen interface thickness parameter type");
      }
      prm.leave_subsection();

      predictor.add_parameters(prm);
      linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  ReinitializationData<number>::post()
  {
    // set the number of initial reinitialization steps equal to the number of reinit steps
    // if no value is provided
    if (n_initial_steps < 0.0)
      n_initial_steps = max_n_steps;

    predictor.post();
  }

  template <typename number>
  void
  ReinitializationData<number>::check_input_parameters(const bool normal_vec_do_matrix_free) const
  {
    AssertThrow(linear_solver.do_matrix_free || implementation == "meltpooldg",
                ExcNotImplemented());
    AssertThrow(normal_vec_do_matrix_free == linear_solver.do_matrix_free,
                ExcMessage("For the reinitialization problem both the "
                           "normal vector and the reinitialization operation have to be "
                           "computed either matrix-based or matrix-free."));
    AssertThrow(interface_thickness_parameter.type ==
                    InterfaceThicknessParameterType::proportional_to_cell_size ||
                  implementation == "meltpooldg",
                ExcMessage("For the adaflo implementation, a variable thickness parameter epsilon "
                           "is mandatory."));
  }

  template struct ReinitializationData<double>;
} // namespace MeltPoolDG::LevelSet
