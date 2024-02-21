#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/level_set/level_set_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  void
  LevelSetData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("levelset");
    {
      prm.add_parameter("ls do reinitialization",
                        do_reinitialization,
                        "Defines if reinitialization of level set function is activated");
      prm.add_parameter(
        "ls n initial reinit steps",
        n_initial_reinit_steps,
        "Defines the number of initial reinitialization steps of the level set function.");
      prm.add_parameter("ls reinit time step size",
                        reinit_time_step_size,
                        "Defines the time step size of the reinitialization.");
      prm.add_parameter("ls time integration scheme",
                        time_integration_scheme,
                        "Determines the time integration scheme.",
                        dealii::Patterns::Selection(
                          "explicit_euler|implicit_euler|crank_nicolson"));
      prm.add_parameter(
        "ls do curvature correction",
        do_curvature_correction,
        "Set this parameter to true if in areas outside the interface region a correction "
        "of the curvature values should be applied. This parameter can be helpful to avoid "
        "numerical instabilities.");
      prm.add_parameter("ls implementation",
                        implementation,
                        "Choose the corresponding implementation of the ls operation.",
                        dealii::Patterns::Selection("meltpooldg|adaflo"));
      prm.add_parameter(
        "ls n subdivisions",
        n_subdivisions,
        "Set the number of subdivisions for the finite element of the level set operation.");
      prm.add_parameter("ls do localized heaviside",
                        do_localized_heaviside,
                        "Determine if the heaviside representation of the level set should be "
                        "calculated as a localized function, being exactly 0 and 1 outside of "
                        "the interface region.");

      prm.add_parameter("tol reinit",
                        tol_reinit,
                        "Set the tolerance for reinitialization. If the "
                        "maximum change of the level set field, i.e. ||ΔФ||∞, exceeds the "
                        "tolerance, reinitialization steps will be performed.");
      nearest_point.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  LevelSetData<number>::post(const unsigned int reinit_max_n_steps)
  {
    // set the number of initial reinitialization steps equal to the number of reinit steps
    // if no value is provided
    if (do_reinitialization && n_initial_reinit_steps < 0.0)
      n_initial_reinit_steps = reinit_max_n_steps;
  }

  template <typename number>
  void
  LevelSetData<number>::check_input_parameters(const unsigned int base_degree) const
  {
    /*
     * The level set problem for simplices can only be solved when no subdivision of the
     * finite element is undertaken.
     */
    AssertThrow((n_subdivisions == 1 || base_degree == 1),
                ExcMessage("If you use n_subdivisions for the level set, degree must be 1."));
  }


  template struct LevelSetData<double>;
} // namespace MeltPoolDG::LevelSet