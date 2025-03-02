/**
 * @brief Collection of parameters required by the compressible Navier-Stokes operator.
 */
#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/cut/cut_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/flow/compressible_fluid_material_data.hpp>
#include <string>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>

namespace MeltPoolDG::Flow
{
  BETTER_ENUM(NumericalFluxType,
              char,
              lax_friedrichs_modified,
              lax_friedrichs_exact,
              harten_lax_vanleer)

  BETTER_ENUM(LinearizedConvectiveFluxJumpType, char, analytic, lambda_fd, complete_fd);

  BETTER_ENUM(JacobianType, char, exact, finite_difference);

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

    // gas phase material data
    CompressibleFluidMaterialPhaseData<double> material_data_gas_phase;

    // fluid phase material data
    // (only relevant for two-phase case)
    CompressibleFluidMaterialPhaseData<double> material_data_liquid_phase;

    // evaporation mass flux
    // (TODO: use Hertz-Knudsen theory and enable constant evaporation mass flux for testing)
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
        material_data_gas_phase.add_parameters(prm, true /*is_gas_phase*/);
        material_data_liquid_phase.add_parameters(prm, false /*is_gas_phase*/);
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
      material_data_gas_phase.thermal_conductivity =
        material_data_gas_phase.dynamic_viscosity * material_data_gas_phase.gamma * material_data_gas_phase.specific_gas_constant / (material_data_gas_phase.gamma - 1.) * 1. / 0.71;
      if (cut.two_phase)
        material_data_liquid_phase.thermal_conductivity =
          material_data_liquid_phase.dynamic_viscosity * material_data_liquid_phase.gamma * material_data_liquid_phase.specific_gas_constant / (material_data_liquid_phase.gamma - 1.) * 1. / 0.71;

      // Ensure that parameters are set for advanced EOS
      if (material_data_gas_phase.equation_of_state == EOS::stiffened_gas)
        AssertThrow(!dealii::numbers::is_invalid(material_data_gas_phase.eos_parameters.p_inf),
                  dealii::ExcMessage(
                    "Inverse time step size must be set to compute the rhs vector."));
      else if (material_data_gas_phase.equation_of_state == EOS::noble_abel_stiffend_gas)
        AssertThrow(!dealii::numbers::is_invalid(material_data_gas_phase.eos_parameters.p_inf) &&
                    !dealii::numbers::is_invalid(material_data_gas_phase.eos_parameters.b) &&
                    !dealii::numbers::is_invalid(material_data_gas_phase.eos_parameters.q),
                  dealii::ExcMessage(
                    "Inverse time step size must be set to compute the rhs vector."));

      if (cut.two_phase)
        {
          if (material_data_liquid_phase.equation_of_state == EOS::stiffened_gas)
            AssertThrow(!dealii::numbers::is_invalid(material_data_liquid_phase.eos_parameters.p_inf),
                      dealii::ExcMessage(
                        "Inverse time step size must be set to compute the rhs vector."));
          else if (material_data_liquid_phase.equation_of_state == EOS::noble_abel_stiffend_gas)
            AssertThrow(!dealii::numbers::is_invalid(material_data_liquid_phase.eos_parameters.p_inf) &&
                    !dealii::numbers::is_invalid(material_data_liquid_phase.eos_parameters.b) &&
                    !dealii::numbers::is_invalid(material_data_liquid_phase.eos_parameters.q),
                      dealii::ExcMessage(
                        "Inverse time step size must be set to compute the rhs vector."));
        }

      // Advanced EOS are currently only allowed for explicit time integration.
      if (domain_representation_type=="fitted_mesh" && material_data_gas_phase.equation_of_state != EOS::ideal_gas)
        AssertThrow(
          !MeltPoolDG::time_integrator_scheme_is_explicit(time_integrator.integrator_type),
          ExcMessage(
            "Only the ideal gas EOS is allowed for implicit time integration."));
    }
  };
} // namespace MeltPoolDG::Flow
