/**
 * @brief A collection of functions for the computation of the interface terms for
 * compressible two-phase flows.
 *
 * Two different strategies for enforcing the interface jump conditions are implemented:
 *
 * - strong enforcement in the weak formulation and penalty enforcement for Dirichlet density and
 *   temperature constraint for the gas phase
 * - HLLP0 approximate Riemann solver for convective fluxes and weighted average Nitsche-type method
 *   for viscous fluxes
 */

#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_eos_utils.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <tuple>
#include <utility>

namespace MeltPoolDG::Multiphase
{
  ///////////////////////////////////////////////////////////////////////////////////////////
  ///                               Penalty method                                        ///
  ///////////////////////////////////////////////////////////////////////////////////////////

  /**
   * This function calculates the interface flux terms, considering both convective and viscous
   * interface jump conditions. A combination of strong enforcement within the weak form and a
   * penalty method is applied.
   *
   * @param u_liquid Conserved variables for liquid phase at quadrature point on the (unfitted) interface.
   * @param u_gas Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param grad_u_liquid Gradient of conserved variables for liquid phase at quadrature point on the (unfitted) interface.
   * @param grad_u_gas Gradient of conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   * @param viscous_terms Collection of helper functions for viscous term evaluations.
   *
   * @return Pair of interface fluxes, considering both convective and viscous jump conditions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_interface_flux_penalty(
      const ConservedVariablesType                                        &u_liquid,
      const ConservedVariablesType                                        &u_gas,
      const ConservedVariablesGradType                                    &grad_u_liquid,
      const ConservedVariablesGradType                                    &grad_u_gas,
      const Flow::CompressibleFlowData                                    &flow_data,
      const MeltPoolDG::Flow::CompressibleFlowViscousKernels<dim, number> &viscous_terms)
  {
    // mass conservation
    const auto tmp_mass_1 =
      (u_liquid[0] / u_gas[0] * u_gas[1] - u_gas[0] / u_liquid[0] * u_liquid[1] +
       (u_gas[0] * u_gas[0] - u_liquid[0] * u_liquid[0]) / (u_liquid[0] * u_gas[0]) *
         flow_data.multiphase.m_dot_evap);

    // TODO: investigate weighting factors!
    const VectorizedArray<double> omega_1    = 0.5;
    const VectorizedArray<double> omega_2    = 0.5;
    const auto                    tmp_mass_2 = omega_1 * u_gas[1] + omega_2 * u_liquid[1];

    // penalty approach for density constraint in gas phase
    // TODO: compute density in gas phase from Hertz-Knudsen theory
    const auto tmp_mass_3 = flow_data.multiphase.density_constraint_penalty_factor *
                            (u_gas[0] - flow_data.multiphase.target_density_gas_phase);

    // momentum conservation
    const auto tmp_momentum_1 =
      (u_liquid[1] * u_liquid[1] / u_liquid[0] - u_gas[1] * u_gas[1] / u_gas[0]) -
      (1. / u_liquid[0] - 1. / u_gas[0]) * flow_data.multiphase.m_dot_evap *
        flow_data.multiphase.m_dot_evap;
    const auto grad_vel_liquid =
      Flow::calculate_grad_velocity<dim, number>(u_liquid, grad_u_liquid);
    const auto grad_vel_gas = Flow::calculate_grad_velocity<dim, number>(u_gas, grad_u_gas);
    const auto tau_liquid =
      viscous_terms.template calculate_viscous_stress_tensor<false /*is_gas_phase*/>(
        grad_vel_liquid);
    auto       tau_liquid_tmp = tau_liquid[0][0];
    const auto tau_gas =
      viscous_terms.template calculate_viscous_stress_tensor<true /*is_gas_phase*/>(grad_vel_gas);
    auto tau_gas_tmp = tau_gas[0][0];
    auto tmp_momentum_2 =
      (u_liquid[1] * u_liquid[1] / u_liquid[0] +
       MeltPoolDG::Flow::calculate_pressure<dim, number, false>(u_liquid, flow_data) -
       tau_liquid_tmp) *
      omega_2;
    tmp_momentum_2 +=
      (u_gas[1] * u_gas[1] / u_gas[0] +
       MeltPoolDG::Flow::calculate_pressure<dim, number, true>(u_gas, flow_data) - tau_gas_tmp) *
      omega_1;

    // energy conservation
    const auto vel_liquid = MeltPoolDG::Flow::calculate_velocity<dim, number>(u_liquid);
    const auto vel_gas    = MeltPoolDG::Flow::calculate_velocity<dim, number>(u_gas);
    const VectorizedArray<double> delta_q = 0.;
    auto                          tmp_energy_1 =
      (u_liquid[dim + 1] * u_liquid[1] / u_liquid[0] - u_gas[dim + 1] * u_liquid[1] / u_liquid[0]) -
      flow_data.multiphase.m_dot_evap *
        (u_liquid[dim + 1] / u_liquid[0] - u_gas[dim + 1] / u_gas[0]) +
      delta_q;
    auto tmp_energy_2 =
      u_liquid[dim + 1] * vel_liquid +
      (MeltPoolDG::Flow::calculate_pressure<dim, number, false /*is_gas_phase*/>(u_liquid,
                                                                                 flow_data) -
       tau_liquid_tmp) *
        vel_liquid -
      flow_data.material_data_gas_phase.thermal_conductivity *
        MeltPoolDG::Flow::calculate_grad_T<dim, number, false /*is_gas_phase*/>(u_liquid,
                                                                                grad_u_liquid,
                                                                                flow_data);
    tmp_energy_2 *= omega_2;
    tmp_energy_2 +=
      (u_gas[dim + 1] * vel_gas +
       (MeltPoolDG::Flow::calculate_pressure<dim, number, true /*is_gas_phase*/>(u_gas, flow_data) -
        tau_gas_tmp) *
         vel_gas -
       flow_data.material_data_liquid_phase.thermal_conductivity *
         MeltPoolDG::Flow::calculate_grad_T<dim, number, true /*is_gas_phase*/>(u_gas,
                                                                                grad_u_gas,
                                                                                flow_data)) *
      omega_1;

    // penalty approach for temperature_2 constraint
    const auto temperature_gas =
      calculate_temperature<dim, number, true /*is_gas_phase*/>(u_gas, flow_data);
    // TODO: compute temperature in gas phase from Hertz-Knudsen theory
    auto tmp_energy_3 = flow_data.multiphase.temperature_constraint_penalty_factor *
                        (temperature_gas - flow_data.multiphase.target_temperature_gas_phase);

    ConservedVariablesType total_flux_liquid;
    ConservedVariablesType total_flux_gas;

    total_flux_liquid[0] = omega_1 * tmp_mass_1 + tmp_mass_2;
    total_flux_liquid[1] = tmp_momentum_1 * omega_1 + tmp_momentum_2;
    total_flux_liquid[2] = tmp_energy_1 * omega_1 + tmp_energy_2[0][0];

    total_flux_gas[0] = omega_2 * tmp_mass_1 - tmp_mass_2 + tmp_mass_3;
    total_flux_gas[1] = tmp_momentum_1 * omega_2 - tmp_momentum_2;
    total_flux_gas[2] = tmp_energy_1 * omega_2 - tmp_energy_2[0][0] + tmp_energy_3;

    return {total_flux_liquid, total_flux_gas};
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  ///             HLLP0 convective flux/Nitsche-weighted viscous flux method              ///
  ///////////////////////////////////////////////////////////////////////////////////////////

  /**
   * This function calculates the convective fluxes for both phases at the phase interface and the
   * interface normal speed at the considered quadrature point. The HLLP0 approximate Riemannn
   * solver for phase transition is implemented according to the following paper: Joens, Munz, 2023:
   * 'Riemann solvers for phase transition in a compressible sharp-interface method.'
   *
   * @param u_liquid Conserved variables for liquid phase at quadrature point on the (unfitted) interface.
   * @param u_gas Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param m_dot_evap Evaporation mass flux from Hertz-Knudsen theory.
   * @param convective_kernels Collection of convective term computations for the compressible Navier-Stokes equations.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Tuple, containing the convective fluxes for liquid and gas phase, respectively, and the interface normal velocity.
   *
   * @note The HLLP0 solver can be solved explicitely, no iterative solution strategy is required.
   * The solver is also applicable for vanishing evaporation mass flux.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<ConservedVariablesType, ConservedVariablesType, VectorizedArray<number>>
    calculate_convective_interface_flux_HLLP0(
      const ConservedVariablesType                                           &u_liquid,
      const ConservedVariablesType                                           &u_gas,
      const Tensor<1, dim, VectorizedArray<number>>                          &normal,
      const double                                                           &m_dot_evap,
      const MeltPoolDG::Flow::CompressibleFlowConvectiveKernels<dim, number> &convective_kernels,
      const Flow::CompressibleFlowData                                       &flow_data)
  {
    // Note: Variables, that are relevant for both the liquid and the gas phase ,are considered as
    // arrays of length 2 in the following. The first element refers to the liquid phase and the
    // second element to the gas phase.

    // 0) preliminaries

    std::array<ConservedVariablesType, 2>                  u = {{u_liquid, u_gas}};
    std::array<Tensor<1, dim, VectorizedArray<number>>, 2> vel;
    std::array<VectorizedArray<number>, 2>                 pressure;
    std::array<VectorizedArray<number>, 2>                 rho;
    std::array<VectorizedArray<number>, 2>                 rho_E;
    std::array<VectorizedArray<number>, 2>                 speed_of_sound;
    for (unsigned int i : {0, 1})
      {
        vel[i]   = Flow::calculate_velocity<dim>(u[i]);
        rho[i]   = u[i][0];
        rho_E[i] = u[i][dim + 1];
      }
    pressure[0] = Flow::calculate_pressure<dim, number, false /*is_gas_phase*/>(u[0], flow_data);
    pressure[1] = Flow::calculate_pressure<dim, number, true /*is_gas_phase*/>(u[1], flow_data);
    speed_of_sound[0] =
      Flow::calculate_speed_of_sound<dim, number, false /*is_gas_phase*/>(u[0], flow_data);
    speed_of_sound[1] =
      Flow::calculate_speed_of_sound<dim, number, true /*is_gas_phase*/>(u[1], flow_data);

    // 1) project velocity and kinetic energy into normal direction of the interface

    std::array<VectorizedArray<number>, 2> vel_n;
    std::array<VectorizedArray<number>, 2> E_n;
    for (unsigned int i : {0, 1})
      {
        vel_n[i] = vel[i] * normal;
        E_n[i]   = rho_E[i] / rho[i] - 0.5 * (vel[i] * vel[i] - vel_n[i] * vel_n[i]);
      }

    // 2) shock speed estimation

    std::array<VectorizedArray<number>, 2> shock_speed;
    shock_speed[0] = vel_n[0] - speed_of_sound[0];
    shock_speed[1] = vel_n[1] + speed_of_sound[1];

    // 3) calculate helpers using Rankine-Hugoniot conditions

    std::array<VectorizedArray<number>, 2> m_hat;
    std::array<VectorizedArray<number>, 2> I_hat;
    std::array<VectorizedArray<number>, 2> E_hat;
    for (unsigned int i : {0, 1})
      {
        m_hat[i] = rho[i] * (vel_n[i] - shock_speed[i]);
        I_hat[i] = m_hat[i] * vel_n[i] + pressure[i];
        E_hat[i] = m_hat[i] * E_n[i] + pressure[i] * vel_n[i];
      }

    // 4) calculate intermediate velocity states

    // TODO: consider surface tension for dim>1 here
    const VectorizedArray<double> delta_p = 0.;

    // TODO: consider Hertz-Knudsen theory for evaporation mass flux here
    std::array<VectorizedArray<number>, 2> tmp_1;
    std::array<VectorizedArray<number>, 2> tmp_2;
    for (unsigned int i : {0, 1})
      {
        tmp_1[i] = m_dot_evap / m_hat[i];
        tmp_2[i] = m_dot_evap - m_hat[i];
      }

    std::array<VectorizedArray<number>, 2> numerator;
    std::array<VectorizedArray<number>, 2> denominator;

    numerator[0] =
      tmp_2[1] * (tmp_1[0] * shock_speed[0] - tmp_1[1] * shock_speed[1]) / (1. - tmp_1[1]) -
      I_hat[0] - delta_p + I_hat[1];
    numerator[1] =
      tmp_2[0] * (tmp_1[1] * shock_speed[1] - tmp_1[0] * shock_speed[0]) / (1. - tmp_1[0]) -
      I_hat[1] + delta_p + I_hat[0];
    denominator[0] = tmp_2[0] - (1. - tmp_1[0]) / (1. - tmp_1[1]) * tmp_2[1];
    denominator[1] = tmp_2[1] - (1. - tmp_1[1]) / (1. - tmp_1[0]) * tmp_2[0];

    std::array<VectorizedArray<number>, 2> vel_n_star;
    for (unsigned int i : {0, 1})
      vel_n_star[i] = numerator[i] / denominator[i];

    // 5) calculate intermediate pressure

    std::array<VectorizedArray<number>, 2> pressure_star;
    for (unsigned int i : {0, 1})
      pressure_star[i] = I_hat[i] - m_hat[i] * vel_n_star[i];

    // 6) re-project normal velocity to Cartesian coordinates

    std::array<Tensor<1, dim, VectorizedArray<number>>, 2> vel_star_cartesian;

    if (dim == 1)
      {
        for (unsigned int i : {0, 1})
          vel_star_cartesian[i] = vel_n_star[i] * normal;
      }
    else
      {
        // TODO: check for dim=3
        Tensor<1, dim, VectorizedArray<number>> tangential_vector;
        tangential_vector[0] = normal[1];
        tangential_vector[1] = -normal[0];

        for (unsigned int i : {0, 1})
          vel_star_cartesian[i] =
            vel_n_star[i] * normal + (vel[i] * tangential_vector) * tangential_vector;
      }

    // 7) calculate conservative variable state vectors of inner states

    std::array<ConservedVariablesType, 2> u_star;
    for (unsigned int i : {0, 1})
      {
        u_star[i][0] = m_hat[i] / (vel_n_star[i] - shock_speed[i]);
        for (unsigned int j = 1; j < dim + 1; j++)
          u_star[i][j] = u_star[i][0] * vel_star_cartesian[i][j];
        u_star[i][dim + 1] =
          (E_hat[i] - pressure_star[i] * vel_n_star[i]) / (vel_n_star[i] - shock_speed[i]) -
          0.5 * u_star[i][0] * vel_n_star[i] * vel_n_star[i] +
          0.5 * u_star[i][0] * vel_star_cartesian[i] * vel_star_cartesian[i];
      }

    // 8) calculate phase interface velocity

    VectorizedArray<number> numerator_normal_vel =
      vel_n_star[0] * u_star[0][0] - vel_n_star[1] * u_star[1][0];
    VectorizedArray<number> denominator_normal_vel = u_star[0][0] - u_star[1][0];
    // avoid division by zero
    VectorizedArray<number> normal_velocity_interface =
      compare_and_apply_mask<SIMDComparison::greater_than>(std::abs(denominator_normal_vel),
                                                           1.e-12,
                                                           numerator_normal_vel /
                                                             denominator_normal_vel,
                                                           0.);

    // 9) calculate fluxes for the two phases

    std::array<ConservedVariablesType, 2>     flux;
    std::array<ConservedVariablesGradType, 2> conv_flux;

    conv_flux[0] =
      convective_kernels.template calculate_convective_flux<false /*is_gas_phase*/>(u[0]);
    conv_flux[1] =
      convective_kernels.template calculate_convective_flux<true /*is_gas_phase*/>(u[1]);
    flux[0] =
      Flow::contract_tensor_with_normal(conv_flux[0], normal) + shock_speed[0] * (u_star[0] - u[0]);
    flux[1] =
      Flow::contract_tensor_with_normal(conv_flux[1], normal) + shock_speed[1] * (u_star[1] - u[1]);

    flux[0][dim + 1] = 0.;
    flux[1][dim + 1] = 0.;

    return {flux[0], flux[1], normal_velocity_interface};
  }

  /**
   * This function calculates the Dirichlet jump in conservative variables via transformation into
   * primitive variables, as described in: Henneaux, 2023: 'Higher-order enforcement of jumps
   * conditions between compressible viscous phases: An extended interior penalty discontinuous
   * Galerkin method for sharp interface simulation.'
   *
   * @param u_liquid_cons Conserved variables for liquid phase at quadrature point on the (unfitted) interface.
   * @param u_gas_cons Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param m_dot_evap Evaporation mass flux from Hertz-Knudsen theory.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Dirichlet jump for conserved quantities in conservative variable formulation.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType>
  inline DEAL_II_ALWAYS_INLINE //
    ConservedVariablesType
    calculate_Dirichlet_jump_in_conservative_variables(const ConservedVariablesType &u_liquid_cons,
                                                       const ConservedVariablesType &u_gas_cons,
                                                       const double                 &m_dot_evap,
                                                       const Flow::CompressibleFlowData &flow_data)
  {
    auto u_liquid_prim =
      convert_conservative_into_primitive_variables<dim, number, false /*is_gas_phase*/>(
        u_liquid_cons, flow_data);

    auto u_gas_prim =
      convert_conservative_into_primitive_variables<dim, number, true /*is_gas_phase*/>(u_gas_cons,
                                                                                        flow_data);

    // TODO: consider surface tension here
    const VectorizedArray<number> delta_p = 0.;

    // TODO: consider temperature jump according to Hertz-Knudsen theory here
    const VectorizedArray<number> delta_T = 0.;

    // TODO: extend to general case dim>1
    ConservedVariablesType J_Dir;

    // TODO: consider evaporation mass flux from Hertz-Knudsen theory here
    J_Dir[0] = delta_p - m_dot_evap * (u_liquid_prim[1] - u_gas_prim[1]);
    J_Dir[1] = m_dot_evap * (1. / u_liquid_prim[0] - 1. / u_gas_prim[0]);
    J_Dir[2] = delta_T;

    u_liquid_prim = u_gas_prim + J_Dir;
    u_gas_prim    = u_liquid_prim - J_Dir;

    const auto u_liquid_cons_star =
      convert_primitive_into_conservative_variables<dim, number, false /*is_gas_phase*/>(
        u_liquid_prim, flow_data);

    const auto u_gas_cons_star =
      convert_primitive_into_conservative_variables<dim, number, true /*is_gas_phase*/>(u_gas_prim,
                                                                                        flow_data);

    J_Dir = u_liquid_cons_star - u_gas_cons_star;

    return J_Dir;
  }

  /**
   * This function calculates the viscous interface flux, as described in:
   * Henneaux, 2023: 'Higher-order enforcement of jumps conditions between compressible viscous
   * phases: An extended interior penalty discontinuous Galerkin method for sharp interface
   * simulation.'
   *
   * @param u_liquid Conserved variables for liquid phase at quadrature point on the (unfitted) interface.
   * @param u_gas Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param grad_u_liquid Gradient of conserved variables for liquid phase at quadrature point on the (unfitted) interface.
   * @param grad_u_gas Gradient of conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param alpha_1 Weighting factor for Nitsche-type weighted viscous interface fluxes.
   * @param alpha_2 Weighting factor for Nitsche-type weighted viscous interface fluxes.
   * @param m_dot_evap Evaporation mass flux from Hertz-Knudsen theory.
   * @param tau Symmetric interior penalty parameter.
   * @param viscous_terms Collection of helper functions for viscous term evaluations.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the test functions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_viscous_interface_flux(
      const ConservedVariablesType                                        &u_liquid,
      const ConservedVariablesType                                        &u_gas,
      const ConservedVariablesGradType                                    &grad_u_liquid,
      const ConservedVariablesGradType                                    &grad_u_gas,
      const Tensor<1, dim, VectorizedArray<number>>                       &normal,
      const double                                                         alpha_1,
      const double                                                         alpha_2,
      const double                                                         m_dot_evap,
      const double                                                         tau,
      const MeltPoolDG::Flow::CompressibleFlowViscousKernels<dim, number> &viscous_terms,
      const Flow::CompressibleFlowData                                    &flow_data)
  {
    // TODO: add contributions for surface tension, interface heat source (laser energy) and
    // Marangoni forces

    const Tensor<1, dim, VectorizedArray<number>> vel_liquid =
      Flow::calculate_velocity<dim>(u_liquid);
    const Tensor<1, dim, VectorizedArray<number>> vel_gas = Flow::calculate_velocity<dim>(u_gas);

    const VectorizedArray<number> vel_n_liquid = vel_liquid * normal;
    const VectorizedArray<number> vel_n_gas    = vel_gas * normal;

    const auto pressure_liquid = Flow::calculate_pressure<dim, number, true>(u_liquid, flow_data);
    const auto pressure_p      = Flow::calculate_pressure<dim, number, false>(u_gas, flow_data);

    const auto grad_vel_liquid = Flow::calculate_grad_velocity(u_liquid, grad_u_liquid);
    const auto grad_vel_p      = Flow::calculate_grad_velocity(u_gas, grad_u_gas);

    const Tensor<2, dim, VectorizedArray<number>> stress_tensor_liquid =
      viscous_terms.template calculate_viscous_stress_tensor<false /*is_gas_phase*/>(
        grad_vel_liquid);
    const Tensor<2, dim, VectorizedArray<number>> stress_tensor_p =
      viscous_terms.template calculate_viscous_stress_tensor<true /*is_gas_phase*/>(grad_vel_p);

    // compute Robin-type viscous interface jump conditions

    ConservedVariablesType J_Rob;

    // TODO: Add entries for case dim>1
    const VectorizedArray<double> delta_q = 0.;
    J_Rob[dim + 1] = (stress_tensor_liquid * vel_liquid - stress_tensor_p * vel_gas) * normal -
                     (pressure_liquid * vel_n_liquid - pressure_p * vel_n_gas) -
                     m_dot_evap * (u_liquid[dim + 1] / u_liquid[0] - u_gas[dim + 1] / u_gas[0]) +
                     delta_q;

    const ConservedVariablesGradType viscous_flux_liquid =
      viscous_terms.template calculate_viscous_flux<false /*is_gas_phase*/>(u_liquid,
                                                                            grad_u_liquid);

    const ConservedVariablesGradType viscous_flux_gas =
      viscous_terms.template calculate_viscous_flux<true /*is_gas_phase*/>(u_gas, grad_u_gas);

    ConservedVariablesGradType total_flux_liquid = UtilityFunctions::dyadic_product(J_Rob, normal);
    total_flux_liquid += viscous_flux_gas;
    total_flux_liquid *= alpha_1;
    total_flux_liquid += alpha_2 * viscous_flux_liquid;

    ConservedVariablesGradType total_flux_gas =
      UtilityFunctions::dyadic_product(J_Rob, -normal); // opposite normal direction for phase 2
    total_flux_gas += viscous_flux_liquid;
    total_flux_gas *= alpha_2;
    total_flux_gas += alpha_1 * viscous_flux_gas;

    // penalty term

    const auto J_Dir_cons =
      calculate_Dirichlet_jump_in_conservative_variables<dim, number, ConservedVariablesType>(
        u_liquid, u_gas, m_dot_evap, flow_data);

    const auto u_liquid_star = alpha_1 * u_liquid + alpha_2 * (u_gas + J_Dir_cons);
    const auto u_gas_star    = alpha_2 * u_gas + alpha_1 * (u_liquid - J_Dir_cons);

    ConservedVariablesGradType penalty_flux_liquid;
    const auto                 tmp_m = u_liquid - (u_gas_star + J_Dir_cons);
    penalty_flux_liquid              = UtilityFunctions::dyadic_product(tmp_m, normal);
    penalty_flux_liquid *= tau;

    ConservedVariablesGradType penalty_flux_gas;
    const auto                 tmp_p = u_gas - (u_liquid_star - J_Dir_cons);
    penalty_flux_gas                 = UtilityFunctions::dyadic_product(tmp_p, -normal);
    penalty_flux_gas *= tau;

    total_flux_liquid -= penalty_flux_liquid;
    total_flux_gas -= penalty_flux_gas;

    return {Flow::contract_tensor_with_normal(total_flux_liquid, normal),
            Flow::contract_tensor_with_normal(total_flux_gas, normal)};
  }

  /**
   * This function calculates the viscous interface fluxes, which are weighted with the gradient of
   * the test functions, as described in: Henneaux, 2023: 'Higher-order enforcement of jumps
   * conditions between compressible viscous phases: An extended interior penalty discontinuous
   * Galerkin method for sharp interface simulation.'
   *
   * @param u_liquid Conserved variables for liquid phase at quadrature point on the (unfitted) interface.
   * @param u_gas Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param alpha_1 Weighting factor for Nitsche-type weighted viscous interface fluxes.
   * @param alpha_2 Weighting factor for Nitsche-type weighted viscous interface fluxes.
   * @param m_dot_evap Evaporation mass flux from Hertz-Knudsen theory.
   * @param viscous_terms Collection of helper functions for viscous term evaluations.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the gradient of the test
   * functions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesGradType, ConservedVariablesGradType>
    calculate_viscous_interface_flux_gradient(
      const ConservedVariablesType                                        &u_liquid,
      const ConservedVariablesType                                        &u_gas,
      const Tensor<1, dim, VectorizedArray<number>>                       &normal,
      const double                                                         alpha_1,
      const double                                                         alpha_2,
      const double                                                         m_dot_evap,
      const MeltPoolDG::Flow::CompressibleFlowViscousKernels<dim, number> &viscous_terms,
      const Flow::CompressibleFlowData                                    &flow_data)
  {
    const auto J_Dir_cons =
      calculate_Dirichlet_jump_in_conservative_variables<dim, number, ConservedVariablesType>(
        u_liquid, u_gas, m_dot_evap, flow_data);

    const auto u_liquid_star = alpha_1 * u_liquid + alpha_2 * (u_gas + J_Dir_cons);
    const auto u_p_star      = alpha_2 * u_gas + alpha_1 * (u_liquid - J_Dir_cons);

    auto tmp_liquid = u_liquid_star - u_liquid;
    auto tmp_gas    = u_p_star - u_gas;

    ConservedVariablesGradType arg_liquid = UtilityFunctions::dyadic_product(tmp_liquid, normal);
    ConservedVariablesGradType arg_gas    = UtilityFunctions::dyadic_product(tmp_gas, -normal);

    const ConservedVariablesGradType flux_grad_liquid =
      viscous_terms.template calculate_viscous_flux<false /*is_gas_phase*/>(u_liquid, arg_liquid);
    const ConservedVariablesGradType flux_grad_gas =
      viscous_terms.template calculate_viscous_flux<true /*is_gas_phase*/>(u_gas, arg_gas);

    return {flux_grad_liquid, flux_grad_gas};
  }
} // namespace MeltPoolDG::Multiphase
