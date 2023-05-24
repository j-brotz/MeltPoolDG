/* ---------------------------------------------------------------------
 *
 * Author: Tinh Vo, TUM, June 2023
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>

namespace MeltPoolDG
{
  // Choose the problem type, as reflected by the usage of the pseudo-time RTE problem
  BETTER_ENUM(RTEProblemType,
              char,
              // solve for the time-independent plain Radiative Transport Equation
              plain,
              // solve for 1 iteration of pseudo-time dependent Radiative Transport problem as a
              // predictor step, then solve for the Radiative Transport Equation
              time_dependent_predictor,
              // solve for the (pseudo-)time dependent problem,
              // do not solve for the Radiative Transport Equation
              time_dependent_problem)
  // Choose the formulation of the absorptivity coefficient
  BETTER_ENUM(AbsorptivityType,
              char,
              // absorptivity coefficient as material constants
              constant,
              // absorptivity coefficient as a function of heaviside gradient
              gradient_based,
              // TBD absorptivity coefficient as a function of the heaviside gradient,
              // and applying a MacAulay bracket. Fails when run in parallel.
              revised_gradient_based)

  template <typename number = double>
  struct RadiativeTransportData
  {
    RadiativeTransportData()
    {
      linear_solver.solver_type         = LinearSolverType::CG;
      linear_solver.preconditioner_type = PreconditionerType::ILU;
    }
    struct PseudoTimeSteppingData
    {
      PseudoTimeSteppingData()
      {
        linear_solver.solver_type         = LinearSolverType::CG;
        linear_solver.preconditioner_type = PreconditionerType::ILU;
        linear_solver.monitor_type        = LinearSolverMonitorType::none;
      }
      LinearSolverData<number> linear_solver;
      double                   diffusion_term_scaling = 1.;
      double                   advection_term_scaling = 1.;
      double                   time_step_size         = 0.0;
      double                   pseudo_time_scaling    = 0.01;
      int                      max_n_steps            = 1;
      double                   rel_tolerance          = 1e-3;
      void
      add_parameters(dealii::ParameterHandler &prm)
      {
        prm.enter_subsection("pseudo time stepping");
        {
          prm.add_parameter("diffusion term scaling",
                            diffusion_term_scaling,
                            "Scaling parameter of diffusion term.");
          prm.add_parameter("advection term scaling",
                            advection_term_scaling,
                            "Scaling parameter of advection term.");
          prm.add_parameter("time step size",
                            time_step_size,
                            "Sets the step size for pseudo-time stepping.");
          prm.add_parameter(
            "pseudo time scaling",
            pseudo_time_scaling,
            "Determine the pseudo-time step as the product of this scaling and minimum cell size.");
          prm.add_parameter("max n steps", max_n_steps, "Maximum pseudo-time iterations allowed.");
          prm.add_parameter("rel tolerance",
                            rel_tolerance,
                            "Pseudo-time stepping relative tolerance.");
        }
        prm.leave_subsection();
      }
    };
    struct AbsorptivityConstantData
    {
      double absorptivity_gas    = 0.1;
      double absorptivity_liquid = 0.9;
      void
      add_parameters(dealii::ParameterHandler &prm)
      {
        prm.enter_subsection("absorptivity");
        {
          prm.add_parameter("absorptivity gas",
                            absorptivity_gas,
                            "Sets the absorptivity of the gas phase.");
          prm.add_parameter("absorptivity liquid",
                            absorptivity_liquid,
                            "Sets the absorptivity of the liquid phase.");
        }
        prm.leave_subsection();
      };
    };
    struct AbsorptivityGradientBasedData
    {
      double avoid_div_zero_constant = 1e-16;
      void
      add_parameters(dealii::ParameterHandler &prm)
      {
        prm.enter_subsection("absorptivity");
        {
          prm.add_parameter("avoid div zero constant",
                            avoid_div_zero_constant,
                            "Sets the absorptivity of the gas phase.");
        }
        prm.leave_subsection();
      };
    };
    LinearSolverData<number>      linear_solver;
    PseudoTimeSteppingData        pseudo_time_stepping;
    AbsorptivityConstantData      absorptivity_constant_data;
    AbsorptivityGradientBasedData absorptivity_gradient_based_data;

    unsigned int        verbosity_level   = 0;
    RTEProblemType      problem_type      = RTEProblemType::time_dependent_predictor;
    AbsorptivityType    absorptivity_type = AbsorptivityType::revised_gradient_based;
    std::vector<double> laser_direction;
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("rte");
      {
        prm.add_parameter(
          "verbosity level",
          verbosity_level,
          "Sets the maximum verbosity level of the console output. The maximum level with respect to the "
          " base value is decisive.");
        prm.add_parameter(
          "problem type",
          problem_type,
          "Chooses whether to solve for the time-independent Radiative Transport Equation, the pseudo-time dependent Radiative Transport problem, or a combination of both.");
        prm.add_parameter("absorptivity type",
                          absorptivity_type,
                          "Chooses the formulation of the absorptivity coefficient");
        prm.add_parameter("laser direction",
                          laser_direction,
                          "Sets the laser source direction vector.");
        linear_solver.add_parameters(prm);
        pseudo_time_stepping.add_parameters(prm);
        absorptivity_constant_data.add_parameters(prm);
        absorptivity_gradient_based_data.add_parameters(prm);
      }
      prm.leave_subsection();
    };
  };
} // namespace MeltPoolDG