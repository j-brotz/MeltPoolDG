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
#include <meltpooldg/utilities/dealii_tensor.hpp>
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
   * @param u_liquid Conserved variables for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param u_gas Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param grad_u_liquid Gradient of conserved variables for liquid phase at quadrature point on
   * the (unfitted) interface.
   * @param grad_u_gas Gradient of conserved variables for gas phase at quadrature point on the
   * (unfitted) interface.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   * @param viscous_terms_liquid Collection of helper functions for viscous term evaluations in the
   * liquid phase.
   * @param viscous_terms_gas Collection of helper functions for viscous term evaluations in the
   * gas phase.
   *
   * @return Pair of interface fluxes, considering both convective and viscous jump conditions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_convective_and_viscous_interface_flux_penalty(
      const ConservedVariablesType             &u_liquid,
      const ConservedVariablesType             &u_gas,
      const ConservedVariablesGradType         &grad_u_liquid,
      const ConservedVariablesGradType         &grad_u_gas,
      const Flow::CompressibleFlowData<number> &flow_data,
      const auto                               &viscous_terms_liquid,
      const auto                               &viscous_terms_gas)
  {
    AssertThrow(dim == 1,
                dealii::ExcNotImplemented(
                  "Currently, only dim=1 is enabled for "
                  "enforcing interface jump conditions with the penalty method."));

    // enumeration for conserved variables component indices
    using Idx = std::conditional_t<
      dim == 1,
      Flow::Idx1D,
      std::conditional_t<dim == 2, Flow::Idx2D, std::conditional_t<dim == 3, Flow::Idx3D, void>>>;

    ConservedVariablesType total_flux_liquid;
    ConservedVariablesType total_flux_gas;

    // TODO: investigate weighting factors!
    const dealii::VectorizedArray<number> omega_1 = 0.5;
    const dealii::VectorizedArray<number> omega_2 = 0.5;

    ///////////////////////
    // mass conservation //
    ///////////////////////

    const dealii::VectorizedArray<number> interface_mass_flux_conservation_term =
      (u_liquid[Idx::density] / u_gas[Idx::density] * u_gas[Idx::momentum_x] -
       u_gas[Idx::density] / u_liquid[Idx::density] * u_liquid[Idx::momentum_x] +
       (u_gas[Idx::density] * u_gas[Idx::density] -
        u_liquid[Idx::density] * u_liquid[Idx::density]) /
         (u_liquid[Idx::density] * u_gas[Idx::density]) *
         flow_data.interface_jump_conditions.m_dot_evap);

    total_flux_liquid[Idx::density] = omega_1 * interface_mass_flux_conservation_term;
    total_flux_gas[Idx::density]    = omega_2 * interface_mass_flux_conservation_term;

    const dealii::VectorizedArray<number> weighted_average_momentum =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(omega_1,
                                                                    u_gas[Idx::momentum_x],
                                                                    omega_2,
                                                                    u_liquid[Idx::momentum_x]);

    total_flux_liquid[Idx::density] += weighted_average_momentum;
    total_flux_gas[Idx::density] -= weighted_average_momentum;

    // penalty approach for density constraint in gas phase
    // TODO: compute target density in gas phase from Hertz-Knudsen theory
    const dealii::VectorizedArray<number> penalty_gas_density =
      flow_data.interface_jump_conditions.penalty.coefficients.density *
      (u_gas[Idx::density] -
       flow_data.interface_jump_conditions.penalty.target_values.density_gas_phase);

    total_flux_gas[Idx::density] += penalty_gas_density;

    ///////////////////////////
    // momentum conservation //
    ///////////////////////////

    const dealii::VectorizedArray<number> jump_momentum_term =
      (u_liquid[Idx::momentum_x] * u_liquid[Idx::momentum_x] / u_liquid[Idx::density] -
       u_gas[Idx::momentum_x] * u_gas[Idx::momentum_x] / u_gas[Idx::density]) -
      (1. / u_liquid[Idx::density] - 1. / u_gas[Idx::density]) *
        flow_data.interface_jump_conditions.m_dot_evap *
        flow_data.interface_jump_conditions.m_dot_evap;

    total_flux_liquid[Idx::momentum_x] = jump_momentum_term * omega_1;
    total_flux_gas[Idx::momentum_x]    = jump_momentum_term * omega_2;

    // compute stress tensor (pressure and viscous contributions) and convert to type
    // dealii::VectorizedArray<number>
    const dealii::VectorizedArray<number> stress_tensor_liquid =
      Flow::EOS::calculate_stress_tensor<dim, number, false /*is_gas_phase*/>(
        u_liquid, grad_u_liquid, flow_data, viscous_terms_liquid)[0][0];
    const dealii::VectorizedArray<number> stress_tensor_gas =
      Flow::EOS::calculate_stress_tensor<dim, number, true /*is_gas_phase*/>(
        u_gas, grad_u_gas, flow_data, viscous_terms_gas)[0][0];

    const dealii::VectorizedArray<number> weighted_average_momentum_term_liquid =
      u_liquid[Idx::momentum_x] * u_liquid[Idx::momentum_x] / u_liquid[Idx::density] -
      stress_tensor_liquid;
    const dealii::VectorizedArray<number> weighted_average_momentum_term_gas =
      u_gas[Idx::momentum_x] * u_gas[Idx::momentum_x] / u_gas[Idx::density] - stress_tensor_gas;
    const dealii::VectorizedArray<number> weighted_average_momentum_term =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(
        omega_2,
        weighted_average_momentum_term_liquid,
        omega_1,
        weighted_average_momentum_term_gas);

    total_flux_liquid[Idx::momentum_x] += weighted_average_momentum_term;
    total_flux_gas[Idx::momentum_x] -= weighted_average_momentum_term;

    /////////////////////////
    // energy conservation //
    /////////////////////////

    // compute velocities and convert to dealii::VectorizedArray<number>
    const dealii::VectorizedArray<number> vel_liquid =
      MeltPoolDG::Flow::calculate_velocity<dim, number>(u_liquid)[0];
    const dealii::VectorizedArray<number> vel_gas =
      MeltPoolDG::Flow::calculate_velocity<dim, number>(u_gas)[0];

    const dealii::VectorizedArray<number> delta_q = 0.;
    const dealii::VectorizedArray<number> jump_energy_term =
      (u_liquid[Idx::energy] * vel_liquid - u_gas[Idx::energy] * vel_gas) -
      flow_data.interface_jump_conditions.m_dot_evap *
        (u_liquid[Idx::energy] / u_liquid[Idx::density] -
         u_gas[Idx::energy] / u_gas[Idx::density]) +
      delta_q;

    total_flux_liquid[Idx::energy] = jump_energy_term * omega_1;
    total_flux_gas[Idx::energy]    = jump_energy_term * omega_2;

    const dealii::VectorizedArray<number> weighted_average_energy_term_liquid =
      u_liquid[Idx::energy] * vel_liquid - stress_tensor_liquid * vel_liquid -
      flow_data.material.gas.thermal_conductivity *
        MeltPoolDG::Flow::EOS::calculate_grad_T<dim, number, false /*is_gas_phase*/>(u_liquid,
                                                                                     grad_u_liquid,
                                                                                     flow_data)[0];
    const dealii::VectorizedArray<number> weighted_average_energy_term_gas =
      u_gas[Idx::energy] * vel_gas - stress_tensor_gas * vel_gas -
      flow_data.material.liquid.thermal_conductivity *
        MeltPoolDG::Flow::EOS::calculate_grad_T<dim, number, true /*is_gas_phase*/>(u_gas,
                                                                                    grad_u_gas,
                                                                                    flow_data)[0];
    const dealii::VectorizedArray<number> weighted_average_energy_term =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(
        omega_2, weighted_average_energy_term_liquid, omega_1, weighted_average_energy_term_gas);

    total_flux_liquid[Idx::energy] += weighted_average_energy_term;
    total_flux_gas[Idx::energy] -= weighted_average_energy_term;

    // penalty approach for gas temperature constraint
    const dealii::VectorizedArray<number> temperature_gas =
      Flow::EOS::calculate_temperature<dim, number, true /*is_gas_phase*/>(u_gas, flow_data);
    // TODO: compute target temperature in gas phase from Hertz-Knudsen theory
    const dealii::VectorizedArray<number> penalty_gas_temperature =
      flow_data.interface_jump_conditions.penalty.coefficients.temperature *
      (temperature_gas -
       flow_data.interface_jump_conditions.penalty.target_values.temperature_gas_phase);

    total_flux_gas[Idx::energy] += penalty_gas_temperature;

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
   * @param u_liquid Conserved variables for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param u_gas Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param convective_terms_liquid Collection of convective term computations for the compressible
   * Navier-Stokes equations in the liquid phase.
   * @param convective_terms_gas Collection of convective term computations for the compressible
   * Navier-Stokes equations in the gas phase.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Tuple, containing the convective fluxes for liquid and gas phase, respectively, and the
   * interface normal velocity.
   *
   * @note The HLLP0 solver can be solved explicitly, no iterative solution strategy is required.
   * The solver is also applicable for vanishing evaporation mass flux.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<ConservedVariablesType, ConservedVariablesType, dealii::VectorizedArray<number>>
    calculate_convective_interface_flux_HLLP0(
      const ConservedVariablesType                                  &u_liquid,
      const ConservedVariablesType                                  &u_gas,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const auto                                                    &convective_terms_liquid,
      const auto                                                    &convective_terms_gas,
      const Flow::CompressibleFlowData<number>                      &flow_data)
  {
    // Note: Variables, that are relevant for both the liquid and the gas phase ,are considered as
    // arrays of length 2 in the following. The first element refers to the liquid phase and the
    // second element to the gas phase.
    constexpr int liquid = 0;
    constexpr int gas    = 1;

    // enumeration for conserved variables component indices
    using Idx = std::conditional_t<
      dim == 1,
      Flow::Idx1D,
      std::conditional_t<dim == 2, Flow::Idx2D, std::conditional_t<dim == 3, Flow::Idx3D, void>>>;

    // 0) preliminaries

    std::array<ConservedVariablesType, 2>                                  u = {{u_liquid, u_gas}};
    std::array<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>, 2> vel;
    std::array<dealii::VectorizedArray<number>, 2>                         pressure;
    std::array<dealii::VectorizedArray<number>, 2>                         rho;
    std::array<dealii::VectorizedArray<number>, 2>                         rho_E;
    std::array<dealii::VectorizedArray<number>, 2>                         speed_of_sound;
    for (unsigned int i : {0, 1})
      {
        vel[i]   = Flow::calculate_velocity<dim>(u[i]);
        rho[i]   = u[i][0];
        rho_E[i] = u[i][dim + 1];
      }
    pressure[liquid] =
      Flow::EOS::calculate_thermodynamic_pressure<dim, number, false /*is_gas_phase*/>(u[liquid],
                                                                                       flow_data);
    pressure[gas] =
      Flow::EOS::calculate_thermodynamic_pressure<dim, number, true /*is_gas_phase*/>(u[gas],
                                                                                      flow_data);
    speed_of_sound[liquid] =
      Flow::EOS::calculate_speed_of_sound<dim, number, false /*is_gas_phase*/>(u[liquid],
                                                                               flow_data);
    speed_of_sound[gas] =
      Flow::EOS::calculate_speed_of_sound<dim, number, true /*is_gas_phase*/>(u[gas], flow_data);

    // 1) project velocity and kinetic energy into normal direction of the interface

    std::array<dealii::VectorizedArray<number>, 2> vel_n;
    std::array<dealii::VectorizedArray<number>, 2> E_n;
    for (unsigned int i : {0, 1})
      {
        vel_n[i] = vel[i] * normal;
        E_n[i]   = rho_E[i] / rho[i] - 0.5 * (vel[i] * vel[i] - vel_n[i] * vel_n[i]);
      }

    // 2) shock speed estimation

    std::array<dealii::VectorizedArray<number>, 2> shock_speed;
    shock_speed[liquid] = vel_n[liquid] - speed_of_sound[liquid];
    shock_speed[gas]    = vel_n[gas] + speed_of_sound[gas];

    // 3) calculate helpers using Rankine-Hugoniot conditions

    std::array<dealii::VectorizedArray<number>, 2> m_hat;
    std::array<dealii::VectorizedArray<number>, 2> I_hat;
    std::array<dealii::VectorizedArray<number>, 2> E_hat;
    for (unsigned int i : {0, 1})
      {
        m_hat[i] = rho[i] * (vel_n[i] - shock_speed[i]);
        I_hat[i] = m_hat[i] * vel_n[i] + pressure[i];
        E_hat[i] = m_hat[i] * E_n[i] + pressure[i] * vel_n[i];
      }

    // 4) calculate intermediate velocity states

    // TODO: consider surface tension for dim>1 here
    const dealii::VectorizedArray<number> delta_p = 0.;

    // TODO: consider Hertz-Knudsen theory for evaporation mass flux here
    std::array<dealii::VectorizedArray<number>, 2> tmp_1;
    std::array<dealii::VectorizedArray<number>, 2> tmp_2;
    for (unsigned int i : {0, 1})
      {
        tmp_1[i] = flow_data.interface_jump_conditions.m_dot_evap / m_hat[i];
        tmp_2[i] = flow_data.interface_jump_conditions.m_dot_evap - m_hat[i];
      }

    std::array<dealii::VectorizedArray<number>, 2> numerator;
    std::array<dealii::VectorizedArray<number>, 2> denominator;

    numerator[liquid] = tmp_2[gas] *
                          (tmp_1[liquid] * shock_speed[liquid] - tmp_1[gas] * shock_speed[gas]) /
                          (1. - tmp_1[gas]) -
                        I_hat[liquid] - delta_p + I_hat[gas];
    numerator[gas] = tmp_2[liquid] *
                       (tmp_1[gas] * shock_speed[gas] - tmp_1[liquid] * shock_speed[liquid]) /
                       (1. - tmp_1[liquid]) -
                     I_hat[gas] + delta_p + I_hat[liquid];
    denominator[liquid] = tmp_2[liquid] - (1. - tmp_1[liquid]) / (1. - tmp_1[gas]) * tmp_2[gas];
    denominator[gas]    = tmp_2[gas] - (1. - tmp_1[gas]) / (1. - tmp_1[liquid]) * tmp_2[liquid];

    std::array<dealii::VectorizedArray<number>, 2> vel_n_star;
    for (unsigned int i : {0, 1})
      vel_n_star[i] = numerator[i] / denominator[i];

    // 5) calculate intermediate pressure

    std::array<dealii::VectorizedArray<number>, 2> pressure_star;
    for (unsigned int i : {0, 1})
      pressure_star[i] = I_hat[i] - m_hat[i] * vel_n_star[i];

    // 6) re-project normal velocity to Cartesian coordinates

    std::array<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>, 2> vel_star_cartesian;

    std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> tangent;
    tangent.resize(dim - 1);

    // compute tangential vector for dim=2 and dim=3
    if constexpr (dim == 2)
      {
        tangent[0][0] = normal[1];
        tangent[0][1] = -normal[0];
      }
    else if constexpr (dim == 3)
      {
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> temp_vec;
        temp_vec[0] = 1.;
        // if normal vector is identical with unit vector choose different unit vector to
        // compute the tangent
        dealii::VectorizedArray<number> tolerance = 1.e-10;
        dealii::VectorizedArray<number> norm_diff = (temp_vec - normal).norm();
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> temp_vec_y;
        temp_vec_y[1] = 1.;
        for (int i = 0; i < 3; ++i)
          {
            temp_vec[i] = compare_and_apply_mask<dealii::SIMDComparison::less_than>(norm_diff,
                                                                                    tolerance,
                                                                                    temp_vec_y[i],
                                                                                    temp_vec[i]);
          }
        tangent[0] = temp_vec - (temp_vec * normal) * normal;
        tangent[1] = dealii::cross_product_3d(normal, tangent[0]);
      }

    for (unsigned int i : {0, 1})
      {
        vel_star_cartesian[i] = vel_n_star[i] * normal;
        for (unsigned int j = 0; j < dim - 1; ++j)
          vel_star_cartesian[i] += (vel[i] * tangent[j]) * tangent[j];
      }

    // 7) calculate conservative variable state vectors of inner states

    std::array<ConservedVariablesType, 2> u_star;
    for (unsigned int i : {0, 1})
      {
        u_star[i][Idx::density] = m_hat[i] / (vel_n_star[i] - shock_speed[i]);
        for (unsigned int j = 1; j < dim + 1; j++)
          u_star[i][j] = u_star[i][Idx::density] * vel_star_cartesian[i][j - 1];
        u_star[i][Idx::energy] =
          (E_hat[i] - pressure_star[i] * vel_n_star[i]) / (vel_n_star[i] - shock_speed[i]) -
          0.5 * u_star[i][Idx::density] * vel_n_star[i] * vel_n_star[i] +
          0.5 * u_star[i][Idx::density] * vel_star_cartesian[i] * vel_star_cartesian[i];
      }

    // 8) calculate phase interface velocity

    dealii::VectorizedArray<number> numerator_normal_vel =
      vel_n_star[liquid] * u_star[liquid][Idx::density] -
      vel_n_star[gas] * u_star[gas][Idx::density];
    dealii::VectorizedArray<number> denominator_normal_vel =
      u_star[liquid][Idx::density] - u_star[gas][Idx::density];
    // avoid division by zero
    dealii::VectorizedArray<number> normal_velocity_interface =
      compare_and_apply_mask<dealii::SIMDComparison::greater_than>(std::abs(denominator_normal_vel),
                                                                   1.e-12,
                                                                   numerator_normal_vel /
                                                                     denominator_normal_vel,
                                                                   0.);

    // 9) calculate fluxes for the two phases

    std::array<ConservedVariablesType, 2>     flux;
    std::array<ConservedVariablesGradType, 2> conv_flux;
    std::array<ConservedVariablesType, 2>     shock_flux;

    conv_flux[liquid] = convective_terms_liquid.calculate_convective_flux(u[liquid]);
    conv_flux[gas]    = convective_terms_gas.calculate_convective_flux(u[gas]);

    for (unsigned int i : {0, 1})
      {
        shock_flux[i] = shock_speed[i] * (u_star[i] - u[i]);
        flux[i]       = contract_tensor_with_vector<dim + 2, dim, number>(conv_flux[i], normal);
      }

    const auto zero_vec = dealii::make_vectorized_array(0.);
    const auto one_vec  = dealii::make_vectorized_array(1.);

    flux[liquid] +=
      shock_flux[liquid] * compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                             shock_speed[liquid], zero_vec, one_vec, zero_vec);
    flux[gas] +=
      shock_flux[gas] * compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
                          shock_speed[gas], zero_vec, one_vec, zero_vec);

    return {flux[liquid], flux[gas], normal_velocity_interface};
  }

  /**
   * This function calculates the Dirichlet jump in conservative variables via transformation into
   * primitive variables, as described in: Henneaux, 2023: 'Higher-order enforcement of jumps
   * conditions between compressible viscous phases: An extended interior penalty discontinuous
   * Galerkin method for sharp interface simulation.'
   *
   * @param u_liquid_cons Conserved variables for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param u_gas_cons Conserved variables for gas phase at quadrature point on the (unfitted)
   * interface.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Dirichlet jump for conserved quantities in conservative variable formulation.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType>
  inline DEAL_II_ALWAYS_INLINE //
    ConservedVariablesType
    calculate_Dirichlet_jump_in_conservative_variables(
      const ConservedVariablesType             &u_liquid_cons,
      const ConservedVariablesType             &u_gas_cons,
      const Flow::CompressibleFlowData<number> &flow_data)
  {
    // enumeration for conserved variables component indices
    using Idx = std::conditional_t<
      dim == 1,
      Flow::Idx1D,
      std::conditional_t<dim == 2, Flow::Idx2D, std::conditional_t<dim == 3, Flow::Idx3D, void>>>;

    auto u_liquid_prim =
      Flow::EOS::convert_conservative_into_primitive_variables<dim, number, false /*is_gas_phase*/>(
        u_liquid_cons, flow_data);

    auto u_gas_prim =
      Flow::EOS::convert_conservative_into_primitive_variables<dim, number, true /*is_gas_phase*/>(
        u_gas_cons, flow_data);

    // TODO: consider surface tension here
    const dealii::VectorizedArray<number> delta_p = 0.;

    // TODO: consider temperature jump according to Hertz-Knudsen theory here
    const dealii::VectorizedArray<number> delta_T = 0.;

    // TODO: extend to general case dim>1
    ConservedVariablesType J_Dir;

    // TODO: consider evaporation mass flux from Hertz-Knudsen theory here
    J_Dir[Idx::density] =
      delta_p - flow_data.interface_jump_conditions.m_dot_evap * (u_liquid_prim[1] - u_gas_prim[1]);
    J_Dir[Idx::momentum_x] =
      flow_data.interface_jump_conditions.m_dot_evap * (1. / u_liquid_prim[0] - 1. / u_gas_prim[0]);
    J_Dir[Idx::energy] = delta_T;

    u_liquid_prim = u_gas_prim + J_Dir;
    u_gas_prim    = u_liquid_prim - J_Dir;

    const auto u_liquid_cons_star =
      Flow::EOS::convert_primitive_into_conservative_variables<dim, number, false /*is_gas_phase*/>(
        u_liquid_prim, flow_data);

    const auto u_gas_cons_star =
      Flow::EOS::convert_primitive_into_conservative_variables<dim, number, true /*is_gas_phase*/>(
        u_gas_prim, flow_data);

    J_Dir = u_liquid_cons_star - u_gas_cons_star;

    return J_Dir;
  }

  /**
   * This function calculates the viscous interface flux, as described in:
   * Henneaux, 2023: 'Higher-order enforcement of jumps conditions between compressible viscous
   * phases: An extended interior penalty discontinuous Galerkin method for sharp interface
   * simulation.'
   *
   * @param u_liquid Conserved variables for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param u_gas Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param grad_u_liquid Gradient of conserved variables for liquid phase at quadrature point on
   * the (unfitted) interface.
   * @param grad_u_gas Gradient of conserved variables for gas phase at quadrature point on the
   * (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param alpha_1 Weighting factor for Nitsche-type weighted viscous interface fluxes.
   * @param alpha_2 Weighting factor for Nitsche-type weighted viscous interface fluxes.
   * @param tau Symmetric interior penalty parameter.
   * @param viscous_terms_liquid Collection of helper functions for viscous term evaluations in the
   * liquid phase.
   *  @param viscous_terms_liquid Collection of helper functions for viscous term evaluations in the
   * gas phase.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the
   * test functions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_viscous_interface_flux(
      const ConservedVariablesType                                  &u_liquid,
      const ConservedVariablesType                                  &u_gas,
      const ConservedVariablesGradType                              &grad_u_liquid,
      const ConservedVariablesGradType                              &grad_u_gas,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const number                                                  &alpha_1,
      const number                                                  &alpha_2,
      const number                                                  &tau,
      const auto                                                    &viscous_terms_liquid,
      const auto                                                    &viscous_terms_gas,
      const Flow::CompressibleFlowData<number>                      &flow_data)
  {
    // enumeration for conserved variables component indices
    using Idx = std::conditional_t<
      dim == 1,
      Flow::Idx1D,
      std::conditional_t<dim == 2, Flow::Idx2D, std::conditional_t<dim == 3, Flow::Idx3D, void>>>;

    // TODO: add contributions for surface tension, interface heat source (laser energy) and
    // Marangoni forces

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> vel_liquid =
      Flow::calculate_velocity<dim>(u_liquid);
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> vel_gas =
      Flow::calculate_velocity<dim>(u_gas);

    const dealii::VectorizedArray<number> vel_n_liquid = vel_liquid * normal;
    const dealii::VectorizedArray<number> vel_n_gas    = vel_gas * normal;

    const auto pressure_liquid =
      Flow::EOS::calculate_thermodynamic_pressure<dim, number, true>(u_liquid, flow_data);
    const auto pressure_p =
      Flow::EOS::calculate_thermodynamic_pressure<dim, number, false>(u_gas, flow_data);

    const auto grad_vel_liquid = Flow::calculate_grad_velocity(u_liquid, grad_u_liquid);
    const auto grad_vel_p      = Flow::calculate_grad_velocity(u_gas, grad_u_gas);

    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> stress_tensor_liquid =
      viscous_terms_liquid.calculate_viscous_stress_tensor(grad_vel_liquid);
    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> stress_tensor_p =
      viscous_terms_gas.calculate_viscous_stress_tensor(grad_vel_p);

    // compute Robin-type viscous interface jump conditions

    ConservedVariablesType J_Rob;

    // TODO: Add entries for case dim>1
    const dealii::VectorizedArray<number> delta_q = 0.;
    J_Rob[Idx::energy] = (stress_tensor_liquid * vel_liquid - stress_tensor_p * vel_gas) * normal -
                         (pressure_liquid * vel_n_liquid - pressure_p * vel_n_gas) -
                         flow_data.interface_jump_conditions.m_dot_evap *
                           (u_liquid[Idx::energy] / u_liquid[Idx::density] -
                            u_gas[Idx::energy] / u_gas[Idx::density]) +
                         delta_q;

    const ConservedVariablesGradType viscous_flux_liquid =
      viscous_terms_liquid.calculate_viscous_flux(u_liquid, grad_u_liquid);

    const ConservedVariablesGradType viscous_flux_gas =
      viscous_terms_gas.calculate_viscous_flux(u_gas, grad_u_gas);

    ConservedVariablesGradType total_flux_liquid = dyadic_product(J_Rob, normal);
    total_flux_liquid += viscous_flux_gas;
    total_flux_liquid *= alpha_1;
    total_flux_liquid += alpha_2 * viscous_flux_liquid;

    ConservedVariablesGradType total_flux_gas =
      dyadic_product(J_Rob, -normal); // opposite normal direction for phase 2
    total_flux_gas += viscous_flux_liquid;
    total_flux_gas *= alpha_2;
    total_flux_gas += alpha_1 * viscous_flux_gas;

    // penalty term

    const auto J_Dir_cons =
      calculate_Dirichlet_jump_in_conservative_variables<dim, number, ConservedVariablesType>(
        u_liquid, u_gas, flow_data);

    const auto u_liquid_star = UtilityFunctions::calculate_arithmetic_phase_weighted_average(
      alpha_1, u_liquid, alpha_2, u_gas + J_Dir_cons);
    const auto u_gas_star = UtilityFunctions::calculate_arithmetic_phase_weighted_average(
      alpha_2, u_gas, alpha_1, u_liquid - J_Dir_cons);

    ConservedVariablesGradType penalty_flux_liquid;
    const auto                 tmp_m = u_liquid - (u_gas_star + J_Dir_cons);
    penalty_flux_liquid              = dyadic_product(tmp_m, normal);
    penalty_flux_liquid *= tau;

    ConservedVariablesGradType penalty_flux_gas;
    const auto                 tmp_p = u_gas - (u_liquid_star - J_Dir_cons);
    penalty_flux_gas                 = dyadic_product(tmp_p, -normal);
    penalty_flux_gas *= tau;

    total_flux_liquid -= penalty_flux_liquid;
    total_flux_gas -= penalty_flux_gas;

    return {contract_tensor_with_vector<dim + 2, dim, number>(total_flux_liquid, normal),
            contract_tensor_with_vector<dim + 2, dim, number>(total_flux_gas, normal)};
  }

  /**
   * This function calculates the viscous interface fluxes, which are weighted with the gradient of
   * the test functions, as described in: Henneaux, 2023: 'Higher-order enforcement of jumps
   * conditions between compressible viscous phases: An extended interior penalty discontinuous
   * Galerkin method for sharp interface simulation.'
   *
   * @param u_liquid Conserved variables for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param u_gas Conserved variables for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param alpha_1 Weighting factor for Nitsche-type weighted viscous interface fluxes.
   * @param alpha_2 Weighting factor for Nitsche-type weighted viscous interface fluxes.
   * @param viscous_terms Collection of helper functions for viscous term evaluations.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the
   * gradient of the test functions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesGradType, ConservedVariablesGradType>
    calculate_viscous_interface_flux_gradient(
      const ConservedVariablesType                                  &u_liquid,
      const ConservedVariablesType                                  &u_gas,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const number                                                  &alpha_1,
      const number                                                  &alpha_2,
      const auto                                                    &viscous_terms_liquid,
      const auto                                                    &viscous_terms_gas,
      const Flow::CompressibleFlowData<number>                      &flow_data)
  {
    const auto J_Dir_cons =
      calculate_Dirichlet_jump_in_conservative_variables<dim, number, ConservedVariablesType>(
        u_liquid, u_gas, flow_data);

    const auto u_liquid_star = UtilityFunctions::calculate_arithmetic_phase_weighted_average(
      alpha_1, u_liquid, alpha_2, u_gas + J_Dir_cons);
    const auto u_gas_star = UtilityFunctions::calculate_arithmetic_phase_weighted_average(
      alpha_2, u_gas, alpha_1, u_liquid - J_Dir_cons);

    auto tmp_liquid = u_liquid_star - u_liquid;
    auto tmp_gas    = u_gas_star - u_gas;

    ConservedVariablesGradType arg_liquid = dyadic_product(tmp_liquid, normal);
    ConservedVariablesGradType arg_gas    = dyadic_product(tmp_gas, -normal);

    const ConservedVariablesGradType flux_grad_liquid =
      viscous_terms_liquid.calculate_viscous_flux(u_liquid, arg_liquid);
    const ConservedVariablesGradType flux_grad_gas =
      viscous_terms_gas.calculate_viscous_flux(u_gas, arg_gas);

    return {flux_grad_liquid, flux_grad_gas};
  }
} // namespace MeltPoolDG::Multiphase
