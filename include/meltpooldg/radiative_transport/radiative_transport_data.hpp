/* ---------------------------------------------------------------------
 *
 * Author: Tinh Vo, TUM, June 2023
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/radiative_transport/pseudo_rte_data.hpp>

namespace MeltPoolDG::RadiativeTransport
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
    RadiativeTransportData();

    LinearSolverData<number> linear_solver;
    unsigned int             verbosity_level                    = 0;
    RTEPredictorType         predictor_type                     = RTEPredictorType::none;
    AbsorptivityType         absorptivity_type                  = AbsorptivityType::gradient_based;
    double                   avoid_singular_matrix_absorptivity = 1e-16;

    struct AbsorptivityConstantData
    {
      double absorptivity_gas    = 0.1;
      double absorptivity_liquid = 0.9;

      void
      add_parameters(dealii::ParameterHandler &prm);

    } absorptivity_constant_data;

    struct AbsorptivityGradientBasedData
    {
      double avoid_div_zero_constant = 1e-16;

      void
      add_parameters(dealii::ParameterHandler &prm);

    } absorptivity_gradient_based_data;

    PseudoTimeSteppingData<number> pseudo_time_stepping;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::RadiativeTransport