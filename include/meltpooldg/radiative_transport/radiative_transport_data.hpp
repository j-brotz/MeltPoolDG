/* ---------------------------------------------------------------------
 *
 * Author: Tinh Vo, TUM, June 2023
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/radiative_transport/pseudo_rte_data.hpp>

namespace MeltPoolDG
{
  // Choose the formulation of the absorptivity coefficient
  BETTER_ENUM(AbsorptivityType,
              char,
              // absorptivity coefficient as material constants
              constant,
              // absorptivity coefficient as a function of the heaviside gradient, and applying a
              // MacAulay bracket
              gradient_based)

  BETTER_ENUM(RTEPredictorType,
              char,
              // no predictor specified; use old value as initial guess
              none,
              // use pseude time stepping operation as a predictor
              pseudo_time_stepping)


  template <typename number = double>
  struct RadiativeTransportData
  {
    RadiativeTransportData()
    {
      linear_solver.solver_type         = LinearSolverType::GMRES;
      linear_solver.preconditioner_type = PreconditionerType::ILU;
    }

    LinearSolverData<number> linear_solver;
    unsigned int             verbosity_level   = 0;
    RTEPredictorType         predictor_type    = RTEPredictorType::none;
    AbsorptivityType         absorptivity_type = AbsorptivityType::gradient_based;
    std::vector<double>      laser_direction;
    double                   avoid_singular_matrix_absorptivity = 1e-16;

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

    } absorptivity_constant_data;

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

    } absorptivity_gradient_based_data;

    PseudoTimeSteppingData<number> pseudo_time_stepping;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("rte");
      {
        prm.add_parameter(
          "rte verbosity level",
          verbosity_level,
          "Sets the maximum verbosity level of the console output. The maximum level with respect to the "
          " base value is decisive.");
        prm.add_parameter("predictor type", predictor_type, "Choose a predictor type.");
        prm.add_parameter("absorptivity type",
                          absorptivity_type,
                          "Chooses the formulation of the absorptivity coefficient");
        prm.add_parameter("laser direction",
                          laser_direction,
                          "Sets the laser source direction vector.");
        prm.add_parameter(
          "avoid singular matrix absorptivity",
          avoid_singular_matrix_absorptivity,
          "Minimum value for absorptivity to ensure a non-singular matrix for RTE.");
        linear_solver.add_parameters(prm);
        pseudo_time_stepping.add_parameters(prm);
        absorptivity_constant_data.add_parameters(prm);
        absorptivity_gradient_based_data.add_parameters(prm);
      }
      prm.leave_subsection();
    };
  };
} // namespace MeltPoolDG