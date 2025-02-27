/**
 * @brief Collection of parameters required by the compressible Navier-Stokes operator.
 */
#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/cut/cut_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <string>

namespace MeltPoolDG::Flow
{
  BETTER_ENUM(NumericalFluxType,
              char,
              lax_friedrichs_modified,
              lax_friedrichs_exact,
              harten_lax_vanleer)

  BETTER_ENUM(LinearizedConvectiveFluxJumpType, char, analytic, lambda_fd, complete_fd);

  BETTER_ENUM(JacobianType, char, exact, finite_difference);

  BETTER_ENUM(EquationOfState,
              char,
              ideal_gas,
              stiffened_ideal_gas,
              noble_abel_stiffend_gas)

  BETTER_ENUM(InterfaceNumericalMethod,
              char,
              HLLC0_and_Nitsche,
              penalty)

  struct CompressibleFlowData
  {
    // finite element data
    FiniteElementData fe;

    // time integration data
    TimeIntegratorData time_integrator;

    //TODO: introduce material subgroup

    // ratio of specific heat (specific heat at constant pressure divided by
    // specific heat at constant volume)
    double gamma = 1.4;
    double gamma_2 = 1.4;

    // dynamic viscosity (SI: kg/(m s))
    double dynamic_viscosity = 1.0 / 1600; // set to zero to deactivate viscous effects
    double dynamic_viscosity_2 = 1.0 / 1600;

    // specific gas constant (SI: J/(kg K))
    double specific_gas_constant = 287.1;
    double specific_gas_constant_2 = 287.1;

    // thermal conductivity (SI: W/(m K))
    double thermal_conductivity =
      dynamic_viscosity * gamma * specific_gas_constant / (gamma - 1.) * 1 / 0.71;
    double thermal_conductivity_2 =
      dynamic_viscosity_2 * gamma_2 * specific_gas_constant_2 / (gamma_2 - 1.) * 1 / 0.71;

    // reference density for interior penalty
    double reference_density = 1.0;
    double reference_density_2 = 1.0;

    // equation of state
    EquationOfState equation_of_state = EquationOfState::ideal_gas;
    EquationOfState equation_of_state_2 = EquationOfState::ideal_gas;

    // evaporation mass flux
    double m_dot_evap = 0.;

    // numerical method for interface jump enforcement
    InterfaceNumericalMethod interface_numerical_method = InterfaceNumericalMethod::penalty;

    // density constraint penalty factor
    double density_constraint_penalty_factor = 10.;

    // temperature constraint penalty factor
    double temperature_constraint_penalty_factor = 10.;

    // symmetric interior penalty parameter for the viscous interface term
    double symm_int_penalty_parameter_interface = 1.;

    // numerical flux type for the convective flux
    NumericalFluxType numerical_flux_type = NumericalFluxType::lax_friedrichs_modified;

    // calculation method of the linearized jump operator in the convective numerical flux (required
    // for implicit time stepping). The options are "analytic", "complete_fd" and "lambda_fd"
    LinearizedConvectiveFluxJumpType linearization_jump_convective_flux =
      LinearizedConvectiveFluxJumpType::complete_fd;

    // jacobian approximation type. The options are "exact" and "finite_difference"
    JacobianType jacobian_type = JacobianType::exact;

    // Courant number for the convective time step restriction
    double courant_number = 0.15;

    // similar to Courant number but for the viscous time step restriction
    double viscous_courant_number = 1.0;

    // if set to true the CFL-criteria determines the size of a time step
    bool do_cfl_time_stepping = false;

    // gravity constant used in the body force computation
    double gravity_constant = 0.0;

    // operator type. The options are "fitted" and "cut"
    std::string domain_representation_type = "fitted";

    // verbosity level
    int verbosity_level = -1;

    // cut-related data
    struct Cut
    {
      // consider single-phase or two-phase case?
      bool two_phase = false;

      // flow boundary condition at the immersed boundary
      // (only relevant for single phase flow problem)
      // (choose between: "no_slip_wall", "inflow")
      std::string unfitted_flow_boundary_condition = "no_slip_wall";

      // cut-related stabilization parameters
      CutStabilizationData<double> stabilization;
    } cut;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("compressible navier stokes");
      {
        fe.add_parameters(prm);
        time_integrator.add_parameters(prm);
        prm.add_parameter("gamma", gamma, "Ratio of specific heat (c_p/c_v).");
        prm.add_parameter("gamma 2", gamma_2, "Ratio of specific heat (c_p/c_v).");
        prm.add_parameter("dynamic viscosity", dynamic_viscosity, "Dynamic viscosity.");
        prm.add_parameter("dynamic viscosity 2", dynamic_viscosity_2, "Dynamic viscosity.");
        prm.add_parameter("specific gas constant", specific_gas_constant, "Specific gas constant.");
        prm.add_parameter("specific gas constant 2", specific_gas_constant_2, "Specific gas constant.");
        prm.add_parameter("thermal conductivity", thermal_conductivity, "Thermal conductivity.");
        prm.add_parameter("thermal conductivity 2", thermal_conductivity_2, "Thermal conductivity.");
        prm.add_parameter("reference density",
                          reference_density,
                          "Reference density for computing the interior penalty factor.");
        prm.add_parameter("reference density 2",
                          reference_density_2,
                          "Reference density for computing the interior penalty factor.");
        prm.add_parameter("equation of state",
                          equation_of_state,
                          "Equation of state for phase 1.");
        prm.add_parameter("equation of state 2",
                          equation_of_state_2,
                          "Equation of state for phase 2.");
        prm.add_parameter("evaporation mass flux", m_dot_evap, "Evaporation mass flux.");
        prm.add_parameter("interface numerical method", interface_numerical_method, "Numerical method for enforcing interface jump conditions.");
        prm.add_parameter("density constraint penalty factor", density_constraint_penalty_factor, "Density constraint penalty factor.");
        prm.add_parameter("temperature constraint penalty factor", temperature_constraint_penalty_factor, "Temperature constraint penalty factor.");
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
        prm.add_parameter("domain representation type",
                          domain_representation_type,
                          "Domain representation type. Choose between 'fitted' and 'cut'.",
                          Patterns::Selection("fitted|cut"));
        prm.add_parameter("verbosity level", verbosity_level, "Verbosity level for output.");
        prm.enter_subsection("cut");
        {
          prm.add_parameter("two phase", cut.two_phase, "Is two-phase case?");
          prm.add_parameter("unfitted flow boundary condition",
                            cut.unfitted_flow_boundary_condition,
                            "Flow boundary condition type at the unfitted boundary.");
          cut.stabilization.add_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    void
    post(const FiniteElementData &base_fe_data, const unsigned int base_verbosity_level)
    {
      fe.post(base_fe_data);
      AssertThrow(fe.type == FiniteElementType::FE_DGQ,
                  ExcMessage(
                    "The compressible flow solver only supports elements of type 'FE_DGQ'."));

      // set default time integration scheme for cut
      if (domain_representation_type == "cut")
        {
          if (time_integrator.integrator_type == TimeIntegratorSchemes::not_initialized)
            time_integrator.integrator_type = TimeIntegratorSchemes::explicit_euler;
          AssertThrow(
            time_integrator.integrator_type == TimeIntegratorSchemes::explicit_euler,
            ExcMessage(
              "The cut compressible flow solver only supports explicit Euler time integration."));
        }

      // Synchronize verbosity with base verbosity if not set explicitly.
      if (verbosity_level < 0)
        verbosity_level = base_verbosity_level;

      // For physical consistency, adjust thermal conductivity based on the user-defined dynamic
      // viscosity, gamma and specific gas constant. The Prandtl number = 0.71 is currently set
      // constant.
      thermal_conductivity =
        dynamic_viscosity * gamma * specific_gas_constant / (gamma - 1.) * 1. / 0.71;
      thermal_conductivity_2 =
        dynamic_viscosity_2 * gamma_2 * specific_gas_constant_2 / (gamma_2 - 1.) * 1. / 0.71;
    }
  };
} // namespace MeltPoolDG::Flow
