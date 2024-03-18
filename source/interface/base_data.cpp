#include <meltpooldg/interface/base_data.hpp>

namespace MeltPoolDG
{
  BaseData::BaseData()
  {
    fe.type   = FiniteElementType::FE_Q;
    fe.degree = 1;
  }

  void
  BaseData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("base");
    {
      prm.add_parameter(
        "application name",
        application_name,
        "Sets the base name for the application that will be fed to the problem type.");
      prm.add_parameter("problem name",
                        problem_name,
                        "Sets the base name for the problem that should be solved.");
      prm.add_parameter("dimension", dimension, "Defines the dimension of the problem");
      prm.add_parameter("global refinements",
                        global_refinements,
                        "Defines the number of initial global refinements");
      prm.add_parameter("do print parameters",
                        do_print_parameters,
                        "Set this parameter to true to list parameters in output");
      prm.add_parameter(
        "verbosity level",
        verbosity_level,
        "Sets the maximum verbosity level of the console output. Set this parameter to 0 in case of test files.");

      fe.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  void
  BaseData::check_input_parameters(const unsigned int ls_n_subdivisions) const
  {
    AssertThrow(problem_name != ProblemType::not_initialized,
                ExcMessage("The problem name must be specified!"));
    AssertThrow(application_name != ApplicationName::not_initialized,
                ExcMessage("The application name must be specified!"));

    switch (problem_name)
      {
        case ProblemType::advection_diffusion:
        case ProblemType::reinitialization:
        case ProblemType::heat_transfer:
          AssertThrow(
            ls_n_subdivisions == 1,
            ExcMessage(
              "n_subdivisions for the level set is not supported for your requested problem_type."));
          break;
        default:
          break;
      }
  }
} // namespace MeltPoolDG