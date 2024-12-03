/**
 * @brief Collection of parameters required by the compressible Navier-Stokes operator.
 */
#pragma once

#include <deal.II/base/parameter_handler.h>

#include <string>

namespace MeltPoolDG::Flow
{
  struct CompressibleFlowData
  {
    // ratio of specific heat (specific heat at constant pressure divided by
    // specific heat at constant volume)
    double gamma = 1.4;

    // dynamic viscosity (SI: kg/(m s))
    double dynamic_viscosity = 1.0 / 1600; // set to zero to deactivate viscous effects

    // specific gas constant (SI: J/(kg K))
    double specific_gas_constant = 287;

    // thermal conductivity (SI: W/(m K))
    double thermal_conductivity =
      dynamic_viscosity * gamma * specific_gas_constant / (gamma - 1.) * 1 / 0.71;

    // reference density for interior penalty
    double reference_density = 1.0;

    // numerical flux type for the convective flux
    std::string numerical_flux_type = "lax_friedrichs_modified";

    // calculation method of the linearized jump operator in the convective numerical flux (required
    // for implicit time stepping). The options are "analytic", "complete_fd" and "lambda_fd"
    std::string linearization_jump_convective_flux = "complete_fd";

    // jacobian approximation type. The options are "exact" and "finite_difference"
    std::string jacobian_type = "exact";

    // Courant number for the convective time step restriction
    double courant_number = 0.15;

    // similar to Courant number but for the viscous time step restriction
    double viscous_courant_number = 1.0;

    // ff set to true the CFL-criteria determines the size of a time step
    bool do_cfl_time_stepping = false;

    // gravity constant used in the body force computation
    double gravity_constant = 9.81;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("compressible navier stokes");
      {
        prm.add_parameter("gamma", gamma, "Ratio of specific heat (c_p/c_v).");
        prm.add_parameter("dynamic viscosity", dynamic_viscosity, "Dynamic viscosity.");
        prm.add_parameter("specific gas constant", specific_gas_constant, "Specific gas constant.");
        prm.add_parameter("thermal conductivity", thermal_conductivity, "Thermal conductivity.");
        prm.add_parameter("reference density",
                          reference_density,
                          "Reference density for computing the interior penalty factor.");
        prm.add_parameter(
          "linearization jump convective flux",
          linearization_jump_convective_flux,
          "Calculation method of the linearized jump operator in the convective "
          "numerical flux (required for implicit time stepping). The options are \"analytic\","
          " \"complete_fd\" and \"lambda_fd\".");
        prm.add_parameter("numerical flux type",
                          numerical_flux_type,
                          "Type of the numerical flux.");
        prm.add_parameter("courant number",
                          courant_number,
                          "Courant number for convective time-step limit.");
        prm.add_parameter("viscous courant number",
                          viscous_courant_number,
                          "Characteristic Courant-like number for viscous time-step limit.");

        prm.add_parameter("jacobian type",
                          jacobian_type,
                          "Type of the jacobian. Choose between 'exact' and 'finite_difference'.");
        prm.add_parameter(
          "use cfl criteria",
          do_cfl_time_stepping,
          "If set to true, the CFL time step size criteria is used to determine the time step"
          " size in each iteration.");
        prm.add_parameter("gravity constant", gravity_constant, "Gravity constant.");
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG::Flow
