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
      fe.add_parameters(prm);

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

      prm.enter_subsection("Discontinous Galerkin");
      {
        prm.add_parameter("factor diffusivity",
                          factor_diffusivity,
                          "Set the factor for diffusivity ");

        prm.add_parameter("IP diffusion",
                          IP_diffusion,
                          "Set the internal penalty for diffusivity ");

        prm.add_parameter("use IMEX", use_IMEX, "Set the flag if IMEX integration should be used ");

        prm.add_parameter(
          "use const gradient in RI",
          use_const_gradient_in_RI,
          "Set if the Godunov gradient should be updated every reinitilization step");


        prm.add_parameter("do CFL based time stepping",
                          do_CFL_based_time_stepping,
                          "Sets a flag if the time stepping should be based on the CFL condition");

        prm.add_parameter(
          "time integration scheme",
          time_integration_scheme,
          "Determines the time integration scheme.",
          Patterns::Selection(
            "explicit_euler|implicit_euler|crank_nicolson|bdf_2|RK_stage_1_order_1|RK_stage_2_order_2|RK_stage_3_order_3|RK_stage_5_order_4|RK_stage_7_order_4|RK_stage_9_order_5"));

        prm.add_parameter(
          "IMEX integration scheme",
          IMEX_integration_scheme,
          "Determines the time integration scheme.",
          Patterns::Selection(
            "explicit_euler|implicit_euler|crank_nicolson|bdf_2|RK_stage_1_order_1|RK_stage_2_order_2|RK_stage_3_order_3|RK_stage_5_order_4|RK_stage_7_order_4|RK_stage_9_order_5"));

        prm.add_parameter("CFL",
                          CFL,
                          "Set a CFL number for the pseudo time stepping in reinitilization. ");
      }
      prm.leave_subsection();



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
  ReinitializationData<number>::post(const FiniteElementData &base_fe_data)
  {
    // set the number of initial reinitialization steps equal to the number of reinit steps
    // if no value is provided
    if (n_initial_steps < 0.0)
      n_initial_steps = max_n_steps;

    predictor.post();
    fe.post(base_fe_data);
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

    // Cross check for DG since for DG only matrix free is supported
    if ((fe.type == FiniteElementType::FE_DGQ) && (!linear_solver.do_matrix_free))
      {
        AssertThrow(false, ExcMessage("For the DG element only matrix free is implemented."));
      }
  }

  template struct ReinitializationData<double>;
} // namespace MeltPoolDG::LevelSet
