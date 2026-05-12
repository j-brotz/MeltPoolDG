#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/level_set/reinitialization_data.hpp>

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
      prm.add_parameter(
        "pseudo time step size",
        pseudo_time_step_size,
        "Sets the reinitialization time step size. By default its computed from the cell size.");
      prm.add_parameter(
        "pseudo time step factor",
        pseudo_time_step_factor,
        "Factor on the reinitialization time step size that is computed from the cell size.");
      prm.add_parameter("max n steps",
                        max_n_steps,
                        "Sets the maximum number of reinitialization steps");
      prm.add_parameter("tolerance",
                        tolerance,
                        "Set the tolerance for reinitialization. If the "
                        "maximum change of the level set field, i.e.  orΔФ or∞, exceeds the "
                        "tolerance, reinitialization steps will be performed.");
      prm.add_parameter("tangential diffusion factor",
                        tangential_diffusion_factor,
                        "Factor that multiplies the normal diffusion "
                        "factor (diffusion length) to obtain the diffusion factor in "
                        "the tangential direction.",
                        dealii::Patterns::Double(0.0));
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
                          reinitilization_DG_specific_data.factor_diffusivity,
                          "Set the factor for diffusivity ");

        prm.add_parameter("IP diffusion",
                          reinitilization_DG_specific_data.IP_diffusion,
                          "Set the internal penalty for diffusivity ");

        prm.add_parameter(
          "use const gradient in RI",
          reinitilization_DG_specific_data.use_const_gradient_in_RI,
          "Set if the Godunov gradient should be updated every reinitilization step");


        prm.add_parameter("do CFL based time stepping",
                          reinitilization_DG_specific_data.do_CFL_based_time_stepping,
                          "Sets a flag if the time stepping should be based on the CFL condition");

        prm.add_parameter(
          "time integration scheme",
          reinitilization_DG_specific_data.time_integration_data.integrator_type,
          "Determines the general time integration scheme for the pseudo time integration of the reinilization equation.");

        prm.add_parameter(
          "IMEX integration scheme",
          reinitilization_DG_specific_data.IMEX_integration_data.integrator_type,
          "If a IMEX integration scheme is specifiead, the integration in pseudo time of the reinilization is done with an Implict-Explicit (IMEX) scheme."
          "this means that the diffusion part is treated with the IMEX integration scheme and the hamiltonian is treated with the general time integration scheme."
          "When choosing an implicit scheme with A-stability larger time steps can be chosen only limited by the stability of the hamiltonian part."
          "This is done, since the diffusion part is the most restrictive part for explicit time integration scheme."
          "If a scheme is set, the time step calculation based on a CFL number assumes an A-stable scheme and only calculates the time step based on the hamiltonian.");

        prm.add_parameter("CFL",
                          reinitilization_DG_specific_data.CFL,
                          "Set a CFL number for the pseudo time stepping in reinitilization. ");

        prm.add_parameter(
          "avoid zero division smoothed signum",
          reinitilization_DG_specific_data.avoid_zero_division_smoothed_signum,
          "Sets a constant to avoid zero division in the computation of the smoothed signum.");

        prm.add_parameter("signum smoothness paramater",
                          reinitilization_DG_specific_data.signum_smoothness_paramater,
                          "Sets the smoothness parameter for the smoothed signum.");

        prm.add_parameter(
          "use directed diffusion stabilization",
          reinitilization_DG_specific_data.use_directed_diffusion_stabilization,
          "Sets a flag if directed diffusion stabilization should be used for reinitilization.");

        prm.add_parameter(
          "hyperbolic weighting function_type",
          reinitilization_DG_specific_data.hyperbolic_weighting_function_type,
          "Sets the type of weighting function for the hyperbolic part of the reinit equation.");

        prm.add_parameter(
          "use spatially constant diffusion",
          reinitilization_DG_specific_data.use_spatially_constant_diffusion,
          "Sets a flag if a spatially constant diffusion should be used for reinitilization.");

        prm.add_parameter(
          "use interface movement penalization",
          reinitilization_DG_specific_data.use_interface_movement_penalization,
          "Sets a flag if a penalization of the interface movement should be used.");

        prm.add_parameter(
          "gradient error time derivative threshold",
          reinitilization_DG_specific_data.gradient_error_time_derivative_threshold,
          "Sets the threshold in the time derivative when a reinit procedure reaches a stationary point");
      }
      prm.leave_subsection();

      prm.enter_subsection("elliptic");
      {
        prm.add_parameter(
          "penalty parameter",
          reinitialization_elliptic_specific_data.penalty_parameter,
          "Penalty parameter for the enforcement of the initial position of the zero "
          "level-set iso-surface during the elliptic reinitialization.",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
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

    AssertThrow(modeltype != ModelType::elliptic,
                dealii::ExcMessage(
                  "The elliptic reinitialization is in development and cannot be used yet."));
  }

  template <typename number>
  void
  ReinitializationData<number>::check_input_parameters(const bool normal_vec_do_matrix_free) const
  {
    AssertThrow(linear_solver.do_matrix_free or implementation == "meltpooldg",
                dealii::ExcNotImplemented());
    AssertThrow(normal_vec_do_matrix_free == linear_solver.do_matrix_free,
                dealii::ExcMessage("For the reinitialization operation both the "
                                   "normal vector and the reinitialization operation have to be "
                                   "computed either matrix-based or matrix-free."));
    AssertThrow(interface_thickness_parameter.type ==
                    InterfaceThicknessParameterType::proportional_to_cell_size or
                  implementation == "meltpooldg",
                dealii::ExcMessage("For the adaflo implementation, a variable thickness "
                                   "parameter epsilon is mandatory."));

    // Cross check for DG since for DG only matrix free is supported
    AssertThrow(fe.type != FiniteElementType::FE_DGQ or linear_solver.do_matrix_free,
                dealii::ExcMessage("For the DG element only matrix free is implemented."));

    AssertThrow(pseudo_time_step_factor > 0.0,
                dealii::ExcMessage("The time step factor must be positive."));

    predictor.check_input_parameters();
  }

  template struct ReinitializationData<double>;
} // namespace MeltPoolDG::LevelSet
