/**
 * @brief Collection of parameters required by the compressible Navier-Stokes operator.
 */
#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/cut/cut_data.hpp>
#include <meltpooldg/flow/compressible_flow_material_data.hpp>
#include <meltpooldg/flow/compressible_multiphase/compressible_multiphase_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/numbers.hpp>

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

  template <typename number>
  struct CompressibleFlowData
  {
    // finite element data
    FiniteElementData fe;

    // time integration data
    TimeIntegration::TimeIntegratorData<number> time_integrator;

    struct Material
    {
      // gas phase material data
      CompressibleFluidMaterialPhaseData<number> gas;
      // fluid phase material data
      // (only relevant for two-phase case)
      CompressibleFluidMaterialPhaseData<number> liquid;
    } material;

    // numerical flux type for the convective flux
    NumericalFluxType numerical_flux_type = NumericalFluxType::lax_friedrichs_modified;

    // calculation method of the linearized jump operator in the convective numerical flux (required
    // for implicit time stepping). The options are "analytic", "complete_fd" and "lambda_fd"
    LinearizedConvectiveFluxJumpType linearization_jump_convective_flux =
      LinearizedConvectiveFluxJumpType::complete_fd;

    // jacobian approximation type. The options are "exact" and "finite_difference"
    JacobianType jacobian_type = JacobianType::exact;

    // Courant number for the convective time step restriction
    number courant_number = 0.15;

    // similar to Courant number but for the viscous time step restriction
    number viscous_courant_number = 1.0;

    // if set to true the CFL-criteria determines the size of a time step
    bool do_cfl_time_stepping = false;

    // gravity constant used in the body force computation
    number gravity_constant = 0.0;

    // operator type. The options are "fitted" and "cut"
    std::string domain_representation_type = "fitted";

    // verbosity level
    int verbosity_level = -1;

    // functions for the initial conditions
    struct InitialConditions
    {
      std::string gas{};
      std::string liquid{};
    } initial_conditions;

    // boundary conditions. The options are: "no_slip_wall", "slip_wall", "inflow",
    // "outflow_fixed_energy", "outflow_fixed_pressure"
    struct BoundaryConditions
    {
      std::string left_boundary_condition   = "no_slip_wall";
      std::string right_boundary_condition  = "no_slip_wall";
      std::string top_boundary_condition    = "no_slip_wall";
      std::string bottom_boundary_condition = "no_slip_wall";
      std::string front_boundary_condition  = "no_slip_wall";
      std::string back_boundary_condition   = "no_slip_wall";
    } boundary_conditions;

    // multiphase-related data
    MeltPoolDG::Multiphase::CompressibleMultiphaseData<number> interface_jump_conditions;

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
      CutStabilizationData<number> stabilization;
    } cut;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("compressible navier stokes");
      {
        fe.add_parameters(prm);
        time_integrator.add_parameters(prm);
        prm.enter_subsection("material");
        material.gas.add_parameters(prm, true /*is_gas_phase*/);
        material.liquid.add_parameters(prm, false /*is_gas_phase*/);
        prm.leave_subsection();
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
                          dealii::Patterns::Selection("fitted|cut"));
        prm.add_parameter("verbosity level", verbosity_level, "Verbosity level for output.");
        prm.enter_subsection("initial conditions");
        {
          prm.add_parameter("gas",
                            initial_conditions.gas,
                            "Initial condition function for gas phase.");
          prm.add_parameter("liquid",
                            initial_conditions.liquid,
                            "Initial condition function for liquid phase.");
        }
        prm.leave_subsection();
        prm.enter_subsection("boundary conditions");
        {
          const std::string boundary_condition_types_doc =
            "The following boundary conditions are supported:\n"
            "\t 'no_slip_wall', 'slip_wall', 'inflow', 'outflow_fixed_energy', "
            "\t 'outflow_fixed_pressure'";
          prm.add_parameter("left boundary condition",
                            boundary_conditions.left_boundary_condition,
                            boundary_condition_types_doc);
          prm.add_parameter("right boundary condition",
                            boundary_conditions.right_boundary_condition,
                            boundary_condition_types_doc);
          prm.add_parameter("top boundary condition",
                            boundary_conditions.top_boundary_condition,
                            boundary_condition_types_doc);
          prm.add_parameter("bottom boundary condition",
                            boundary_conditions.bottom_boundary_condition,
                            boundary_condition_types_doc);
          prm.add_parameter("front boundary condition",
                            boundary_conditions.front_boundary_condition,
                            boundary_condition_types_doc);
          prm.add_parameter("back boundary condition",
                            boundary_conditions.back_boundary_condition,
                            boundary_condition_types_doc);
        }
        prm.leave_subsection();
        interface_jump_conditions.add_parameters(prm);
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
                  dealii::ExcMessage(
                    "The compressible flow solver only supports elements of type 'FE_DGQ'."));

      if (cut.two_phase)
        interface_jump_conditions.post();

      // set default time integration scheme for cut
      if (domain_representation_type == "cut")
        {
          if (time_integrator.integrator_type ==
              TimeIntegration::TimeIntegratorSchemes::not_initialized)
            time_integrator.integrator_type =
              TimeIntegration::TimeIntegratorSchemes::explicit_euler;
          AssertThrow(
            time_integrator.integrator_type ==
              TimeIntegration::TimeIntegratorSchemes::explicit_euler,
            dealii::ExcMessage(
              "The cut compressible flow solver only supports explicit Euler time integration."));
        }

      // Synchronize verbosity with base verbosity if not set explicitly.
      if (verbosity_level < 0)
        verbosity_level = base_verbosity_level;

      // Set thermal conductivity, if not explicitly set by the user.
      // For physical consistency, set thermal conductivity based on the user-defined dynamic
      // viscosity, gamma and specific gas constant. The Prandtl number is currently set
      // constant to Pr=0.71 for the gas phase (air) and to Pr=0.01 for the liquid phase (metal).
      if (material.gas.thermal_conductivity == std::numeric_limits<number>::max())
        material.gas.thermal_conductivity = material.gas.dynamic_viscosity * material.gas.gamma *
                                            material.gas.specific_gas_constant /
                                            (material.gas.gamma - 1.) * 1. / 0.71;
      if (cut.two_phase and
          material.liquid.thermal_conductivity == std::numeric_limits<number>::max())
        material.liquid.thermal_conductivity =
          material.liquid.dynamic_viscosity * material.liquid.gamma *
          material.liquid.specific_gas_constant / (material.liquid.gamma - 1.) * 1. / 0.01;

      // Ensure that parameters are set for advanced equations of state
      if (material.gas.eos_data.type == EquationOfState::stiffened_gas)
        AssertThrow(material.liquid.eos_data.p_inf != std::numeric_limits<number>::max(),
                    dealii::ExcMessage(
                      "The parameter p_inf is required for the stiffened gas EOS."));
      else if (material.gas.eos_data.type == EquationOfState::noble_abel_stiffened_gas)
        AssertThrow(material.liquid.eos_data.p_inf != std::numeric_limits<number>::max() and
                      material.liquid.eos_data.b != std::numeric_limits<number>::max() and
                      material.liquid.eos_data.q != std::numeric_limits<number>::min(),
                    dealii::ExcMessage(
                      "The parameters p_inf, b and q are required for the Noble-Abel stiffened"
                      " gas EOS."));

      if (cut.two_phase)
        {
          if (material.liquid.eos_data.type == EquationOfState::stiffened_gas)
            AssertThrow(material.liquid.eos_data.p_inf != std::numeric_limits<number>::max(),
                        dealii::ExcMessage(
                          "The parameter p_inf is required for the stiffened gas EOS."));
          else if (material.liquid.eos_data.type == EquationOfState::noble_abel_stiffened_gas)
            AssertThrow((material.liquid.eos_data.p_inf != std::numeric_limits<number>::max()) and
                          (material.liquid.eos_data.b != std::numeric_limits<number>::max()) and
                          (material.liquid.eos_data.q != std::numeric_limits<number>::min()),
                        dealii::ExcMessage(
                          "The parameters p_inf, b and q are required for the Noble-Abel stiffened"
                          " gas EOS."));
        }

      // Advanced EOS are currently only allowed for explicit time integration.
      if (material.gas.eos_data.type != EquationOfState::ideal_gas)
        AssertThrow(!MeltPoolDG::TimeIntegration::time_integrator_scheme_is_explicit(
                      time_integrator.integrator_type),
                    dealii::ExcMessage(
                      "Only the ideal gas EOS is allowed for implicit time integration."));
    }
  };
} // namespace MeltPoolDG::Flow
