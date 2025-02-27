/**
 * @brief A collection of helper functions for the computation of the interface integrals for
 * compressible two-phase flows.
 */

#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

namespace MeltPoolDG::Multiphase
{
  /**
   * Kernel of the local cell applier for the right-hand side function. This function computes the
   * cell integral contribution to the right hand side for the quadrature point index and the
   * corresponding FE evaluator.
   *
   * @param evaluator FE-evaluator object reinitialized on the current cell batch.
   * @param q Index of the quadrature point.
   * @param constant_body_force Value of the body force. If the body force is not constant the
   * pointer must be set to nullptr.
   *
   * @return Tuple, containing the flux, weighted with the value of the test function, as first
   * argument, and the flux, weighted with the gradient of the test function, as second argument.
   */
  template <int dim, typename Number, typename ConservedVariablesType, typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_interface_flux_penalty(const ConservedVariablesType &u_m,
                                     const ConservedVariablesType &u_p,
                                     const ConservedVariablesGradType &grad_u_m,
                                     const ConservedVariablesGradType &grad_u_p,
                                     const Flow::CompressibleFlowScratchData<dim, Number> &flow_scratch_data,
                                     const MeltPoolDG::Flow::CompressibleFlowViscousKernels<dim, Number> &viscous_terms)
  {
    // mass conservation
    const auto tmp_1 = (u_m[0]/u_p[0]*u_p[1] - u_p[0]/u_m[0]*u_m[1]+(u_p[0]*u_p[0]-u_m[0]*u_m[0])/(u_m[0]*u_p[0])*flow_scratch_data.flow_data.m_dot_evap);

    const VectorizedArray<double> omega_1 = 0.5;
    const VectorizedArray<double> omega_2 = 0.5;
    const auto tmp_2 = omega_1 * u_p[1] + omega_2 * u_m[1];

    // penalty approach for density_2 constraint
    const auto tmp_5 = flow_scratch_data.flow_data.density_constraint_penalty_factor * (u_p[0] - 1.);

    // momentum conservation
    const auto tmp_3 = (u_m[1] * u_m[1] / u_m[0] - u_p[1] * u_p[1] / u_p[0]) - (1./u_m[0] - 1./u_p[0]) * flow_scratch_data.flow_data.m_dot_evap * flow_scratch_data.flow_data.m_dot_evap;
    const auto grad_vel_m = Flow::calculate_grad_velocity<dim,Number>(u_m, grad_u_m);
    const auto grad_vel_p = Flow::calculate_grad_velocity<dim,Number>(u_p, grad_u_p);
    const auto tau_m = viscous_terms.calculate_viscous_stress_tensor(grad_vel_m);
    auto tau_m_tmp = tau_m[0][0];
    const auto tau_p = viscous_terms.calculate_viscous_stress_tensor(grad_vel_p);
    auto tau_p_tmp = tau_p[0][0];
    auto tmp_4 = (u_m[1] * u_m[1] / u_m[0] + MeltPoolDG::Flow::calculate_pressure<dim,Number>(u_m, flow_scratch_data.flow_data.gamma)-tau_m_tmp) * omega_2;
    tmp_4 += (u_p[1] * u_p[1] / u_p[0] + MeltPoolDG::Flow::calculate_pressure<dim,Number>(u_p, flow_scratch_data.flow_data.gamma_2)-tau_p_tmp) * omega_1;

    // energy conservation
    const auto vel_m = MeltPoolDG::Flow::calculate_velocity<dim,Number>(u_m);
    const auto vel_p = MeltPoolDG::Flow::calculate_velocity<dim,Number>(u_p);
    const VectorizedArray<double> delta_q = 0.;
    auto tmp_6 = (u_m[dim+1] * u_m[1] / u_m[0] - u_p[dim+1] * u_m[1] / u_m[0] ) - flow_scratch_data.flow_data.m_dot_evap * (u_m[dim+1]/u_m[0] - u_p[dim+1]/u_p[0]) + delta_q;
    auto tmp_7 = u_m[dim+1] * vel_m + (MeltPoolDG::Flow::calculate_pressure<dim,Number>(u_m, flow_scratch_data.flow_data.gamma) - tau_m_tmp) * vel_m -
      flow_scratch_data.flow_data.thermal_conductivity * MeltPoolDG::Flow::calculate_grad_T<dim>(u_m, grad_u_m, flow_scratch_data.flow_data.gamma_2, flow_scratch_data.flow_data.specific_gas_constant);
    tmp_7 *= omega_2;
    tmp_7 += (u_p[dim+1] * vel_p + (MeltPoolDG::Flow::calculate_pressure<dim,Number>(u_p, flow_scratch_data.flow_data.gamma_2) - tau_p_tmp) * vel_p -
      flow_scratch_data.flow_data.thermal_conductivity_2 * MeltPoolDG::Flow::calculate_grad_T<dim>(u_p, grad_u_p, flow_scratch_data.flow_data.gamma_2, flow_scratch_data.flow_data.specific_gas_constant_2)) * omega_1;

    // penalty approach for temperature_2 constraint
    const auto temperature_p = (u_p[dim+1] - 0.5 * u_p[0] * vel_p * vel_p) / (flow_scratch_data.flow_data.specific_gas_constant_2/(flow_scratch_data.flow_data.gamma_2 - 1.)*u_p[0]);
    auto tmp_8 = flow_scratch_data.flow_data.temperature_constraint_penalty_factor * (temperature_p - 293.15);

    ConservedVariablesType total_flux_m;
    ConservedVariablesType total_flux_p;

    total_flux_m[0] = omega_1 * tmp_1 + tmp_2;
    total_flux_m[1] = tmp_3 * omega_1 + tmp_4;
    total_flux_m[2] = tmp_6 * omega_1 + tmp_7[0][0];

    total_flux_p[0] = 0.*(omega_2 * tmp_1 - tmp_2 + tmp_5);
    total_flux_p[1] = tmp_3 * omega_2 - tmp_4;
    total_flux_p[2] = tmp_6 * omega_2 - tmp_7[0][0] + tmp_8;

    return {total_flux_m, total_flux_p};
  }


  // provide convective fluxes at phase interface, solve the multiphase Riemann problem with the
  // HLLP0 Riemann solver
  template <int dim, typename Number, typename ConservedVariablesType, typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<ConservedVariablesType, ConservedVariablesType,VectorizedArray<Number>>
    calculate_convective_interface_flux_HLLC(const ConservedVariablesType &u_m,
                                        const ConservedVariablesType &u_p,
                                        const Tensor<1, dim, VectorizedArray<Number>>              &normal,
                                        const double                               gamma_m,
                                        const double gamma_p,
                                        const double m_dot_evap,
                                        const MeltPoolDG::Flow::CompressibleFlowConvectiveKernels<dim, Number> &convective_kernels)
  {
    // 1) compute normal velocities

    const VectorizedArray<Number> vel_n_m = Flow::calculate_velocity<dim>(u_m) * normal;
    const VectorizedArray<Number> vel_n_p = Flow::calculate_velocity<dim>(u_p) * normal;

    // 2) shock speed estimation

    AssertThrow(u_m[0][0] > 0. && u_p[0][0] > 0., ExcMessage("Minimum density must not be zero."));

    const auto pressure_m = Flow::calculate_pressure<dim>(u_m, gamma_m);
    const auto speed_of_sound_m = std::sqrt(gamma_m * pressure_m / u_m[0]);

    const auto pressure_p = Flow::calculate_pressure<dim>(u_p, gamma_p);
    const auto speed_of_sound_p = std::sqrt(gamma_p * pressure_p / u_p[0]);

    const auto s_m = vel_n_m - speed_of_sound_m;
    const auto s_p = vel_n_p + speed_of_sound_p;

    // correct kinetic energy

    const auto E_corrected_m = u_m[dim+1]/u_m[0] - 0.5 * Flow::calculate_velocity<dim>(u_m) * Flow::calculate_velocity<dim>(u_m)
                               + 0.5 * vel_n_m * vel_n_m;
    const auto E_corrected_p = u_p[dim+1]/u_p[0] - 0.5 * Flow::calculate_velocity<dim>(u_p) * Flow::calculate_velocity<dim>(u_p)
                               + 0.5 * vel_n_p * vel_n_p;

    // 3) calculate helpers

    const auto m_m = u_m[0] * (vel_n_m - s_m);
    const auto m_p = u_p[0] * (vel_n_p - s_p);

    const auto I_m = m_m * vel_n_m + pressure_m;
    const auto I_p = m_p * vel_n_p + pressure_p;

    const auto E_m = m_m * E_corrected_m + pressure_m * vel_n_m;
    const auto E_p = m_p * E_corrected_p + pressure_p * vel_n_p;

    // 4) calculate intermediate velocity states

    const VectorizedArray<double> delta_p = 0.;

    const auto vel_star_m = ((m_dot_evap-m_p)*(m_dot_evap/m_m*s_m-m_dot_evap/m_p*s_p)/(1.-m_dot_evap/m_p)+I_p-delta_p-I_m)/
      ((m_dot_evap-m_m)-(1.-m_dot_evap/m_m)/(1.-m_dot_evap/m_p)*(m_dot_evap-m_p));

    const auto vel_star_p = ((m_dot_evap-m_m)*(m_dot_evap/m_p*s_p-m_dot_evap/m_m*s_m)/(1.-m_dot_evap/m_m)-I_p+delta_p+I_m)/
      ((m_dot_evap-m_p)-(1.-m_dot_evap/m_p)/(1.-m_dot_evap/m_m)*(m_dot_evap-m_m));

    // 5) calculate intermediate pressure

    const auto pressure_star_m = I_m - m_m * vel_star_m;
    const auto pressure_star_p = I_p - m_p * vel_star_p;

    // 6) calculate conservative variable state vectors of inner states

    ConservedVariablesType u_m_star;
    ConservedVariablesType u_p_star;

    u_m_star[0] = m_m / (vel_star_m - s_m);
    u_p_star[0] = m_p / (vel_star_p - s_p);

    u_m_star[dim+1] = (E_m - pressure_star_m * vel_star_m)/(vel_star_m - s_m) - 0.5 * u_m_star[0] * vel_star_m * vel_star_m;
    u_p_star[dim+1] = (E_p - pressure_star_p * vel_star_p)/(vel_star_p - s_p) - 0.5 * u_p_star[0] * vel_star_p * vel_star_p;

    // reproject normal momentum to cartesian coordinates

    Tensor<1,dim,VectorizedArray<Number>> vel_star_cartesian_m;
    Tensor<1,dim,VectorizedArray<Number>> vel_star_cartesian_p;

    if (dim==1)
      {
        vel_star_cartesian_m = vel_star_m * normal;
        vel_star_cartesian_p = vel_star_p * normal;
      }
    else
      {
        Tensor<1,dim,VectorizedArray<Number>> tangential_vector;
        tangential_vector[0] = normal[1];
        tangential_vector[1] = -normal[0];

        vel_star_cartesian_m = vel_star_m * normal + (Flow::calculate_velocity<dim>(u_m) * tangential_vector) * tangential_vector;
        vel_star_cartesian_p = vel_star_p * normal + (Flow::calculate_velocity<dim>(u_p) * tangential_vector) * tangential_vector;
      }

    for (unsigned int i = 0; i < dim; i++)
      {
        u_m_star[i+1] = u_m_star[0] * vel_star_cartesian_m[i];
        u_p_star[i+1] = u_p_star[0] * vel_star_cartesian_p[i];
      }

    u_m_star[dim+1] += 0.5 * u_m_star[0] * vel_star_cartesian_m * vel_star_cartesian_m;
    u_p_star[dim+1] += 0.5 * u_p_star[0] * vel_star_cartesian_p * vel_star_cartesian_p;

    // 7) calculate phase interface velocity

    VectorizedArray<double> normal_velocity_interface = 0.;
    // avoid bad cases "0/0" and "x/0"
    if (std::abs(vel_star_m * u_m_star[0] - vel_star_p * u_p_star[0])[0] > 1.e-12 && std::abs(u_m_star[0] - u_p_star[0])[0] > 1.e-12)
      normal_velocity_interface = (vel_star_m * u_m_star[0] - vel_star_p * u_p_star[0]) / (u_m_star[0] - u_p_star[0]);

    // 8) calculate fluxes for the two phases

    ConservedVariablesType flux_m;
    ConservedVariablesType flux_p;

    ConservedVariablesGradType conv_flux_m;
    ConservedVariablesGradType conv_flux_p;

    conv_flux_m = convective_kernels.calculate_convective_flux(u_m);
    conv_flux_p = convective_kernels.calculate_convective_flux(u_p);

    flux_m = Flow::contract_tensor_with_normal(conv_flux_m, normal)+ s_m * (u_m_star - u_m);
    flux_p = Flow::contract_tensor_with_normal(conv_flux_p, normal)+ s_p * (u_p_star - u_p);

    return std::make_tuple(flux_m,flux_p,normal_velocity_interface);
  }


  template <int dim, typename Number, typename ConservedVariablesType, typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    ConservedVariablesType
    convert_into_primitive_variables(const ConservedVariablesType &u_cons,
                                     const double &specific_gas_constant,
                                     const double &gamma,
                                     const double &pressure_inf = 0.)
  {
    ConservedVariablesType u_prim;

    // pressure
    u_prim[0] = Flow::calculate_pressure<dim>(u_cons, gamma);

    // velocity
	for (unsigned int i = 1; i < dim+1; i++)
    	u_prim[i] = u_cons[i] / u_cons[0];

    // temperature
    u_prim[dim+1] = u_prim[0] / (specific_gas_constant * u_cons[0]);

    return u_prim;
  }



  template <int dim, typename Number, typename ConservedVariablesType, typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    ConservedVariablesType
    convert_into_conservative_variables(const ConservedVariablesType &u_prim,
                                     const double &specific_gas_constant,
                                     const double &gamma,
                                     const double &pressure_inf = 0.)
  {
    ConservedVariablesType u_cons;

    // density
    u_cons[0] = u_prim[0] / (specific_gas_constant * u_prim[dim+1]);

    // momentum
    for (unsigned int i=1; i<dim+1; i++)
    	u_cons[i] = u_prim[i] * u_cons[0];

    // total energy
    Tensor<1,dim,VectorizedArray<Number>> vel;
    for (unsigned int i=0; i<dim; i++)
      vel[i] = u_prim[i+1];

    u_cons[dim+1] = u_cons[0] * (specific_gas_constant / (gamma -1.) * u_prim[dim+1] + 0.5 * vel * vel) + pressure_inf;

    return u_cons;
  }



  template <int dim, typename Number, typename ConservedVariablesType, typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    ConservedVariablesType
    calculate_Dirichlet_jump_in_conservative_variables(const ConservedVariablesType &u_m_cons,
                                                       const ConservedVariablesType &u_p_cons,
                                                       const double &specific_gas_constant_phase_1,
                                                       const double &specific_gas_constant_phase_2,
                                     				   const double &gamma_phase_1,
                                                       const double &gamma_phase_2,
                                                       const double &m_dot_evap,
                                                       const double &pressure_inf_phase_1,
                                                       const double &pressure_inf_phase_2)
  {
	  auto u_m_prim = convert_into_primitive_variables<dim, Number, ConservedVariablesType, ConservedVariablesGradType>(u_m_cons,
                                       specific_gas_constant_phase_1,
                                       gamma_phase_1,
                                       pressure_inf_phase_1);

  	auto u_p_prim = convert_into_primitive_variables<dim, Number, ConservedVariablesType, ConservedVariablesGradType>(u_p_cons,
                                     specific_gas_constant_phase_2,
                                     gamma_phase_2,
                                     pressure_inf_phase_2);

    // TODO: extend to general case dim>1
	  ConservedVariablesType J_Dir;

    J_Dir[0] = -m_dot_evap * (u_m_prim[1] - u_p_prim[1]);
    J_Dir[1] = m_dot_evap * (1./u_m_prim[0] - 1./u_p_prim[0]);
    J_Dir[2] = 0.;//-174.1553466;

    u_m_prim = u_p_prim + J_Dir;
    u_p_prim = u_m_prim - J_Dir;

	  const auto u_m_cons_star = convert_into_conservative_variables<dim, Number, ConservedVariablesType, ConservedVariablesGradType>(u_m_prim,
                                     specific_gas_constant_phase_1,
                                     gamma_phase_1,
                                     pressure_inf_phase_1);

  	const auto u_p_cons_star = convert_into_conservative_variables<dim, Number, ConservedVariablesType, ConservedVariablesGradType>(u_p_prim,
                                     specific_gas_constant_phase_2,
                                     gamma_phase_2,
                                     pressure_inf_phase_2);

    J_Dir = u_m_cons_star - u_p_cons_star;

    return J_Dir;
  }



  // provide convective fluxes at phase interface, solve the multiphase Riemann problem with the
  // HLLP0 Riemann solver
  template <int dim, typename Number, typename ConservedVariablesType, typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_viscous_interface_flux(const ConservedVariablesType &u_m,
                                        const ConservedVariablesType &u_p,
                                        const ConservedVariablesGradType &grad_u_m,
                                        const ConservedVariablesGradType &grad_u_p,
                                        const Tensor<1,dim,VectorizedArray<Number>>              &normal,
                                        const auto                               &/*tau*/,
                                        const double                               gamma_m,
                                        const double gamma_p,
                                        const double specific_gas_constant_phase_1,
                                        const double specific_gas_constant_phase_2,
                                        const double alpha_1,
                                        const double alpha_2,
                                        const double m_dot_evap,
                                        const double tau,
                                        const double pressure_inf_phase_1,
                                        const double pressure_inf_phase_2,
                                        const MeltPoolDG::Flow::CompressibleFlowViscousKernels<dim, Number> &viscous_terms)
  {
    // TODO: add contributions for surface tension, interface heat source (laser energy) and Marangoni forces

    const Tensor<1,dim,VectorizedArray<Number>> vel_m = Flow::calculate_velocity<dim>(u_m);
    const Tensor<1,dim,VectorizedArray<Number>> vel_p = Flow::calculate_velocity<dim>(u_p);

    const VectorizedArray<Number> vel_n_m = vel_m * normal;
    const VectorizedArray<Number> vel_n_p = vel_p * normal;

    const auto pressure_m = Flow::calculate_pressure<dim>(u_m, gamma_m);
    const auto pressure_p = Flow::calculate_pressure<dim>(u_p, gamma_p);

    const auto grad_vel_m = Flow::calculate_grad_velocity(u_m, grad_u_m);
    const auto grad_vel_p = Flow::calculate_grad_velocity(u_p, grad_u_p);

    const Tensor<2, dim, VectorizedArray<Number>> stress_tensor_m = viscous_terms.calculate_viscous_stress_tensor(grad_vel_m);
    const Tensor<2, dim, VectorizedArray<Number>> stress_tensor_p = viscous_terms.calculate_viscous_stress_tensor(grad_vel_p);

    // compute Robin-type viscous interface jump conditions

    ConservedVariablesType J_Rob;

    // TODO: Add entries for case dim>1
    const VectorizedArray<double> delta_q = 0.;
    J_Rob[dim+1] = (stress_tensor_m * vel_m - stress_tensor_p * vel_p) * normal
                    - (pressure_m * vel_n_m - pressure_p * vel_n_p)
                    - m_dot_evap * (u_m[dim+1]/u_m[0] - u_p[dim+1]/u_p[0])
                    + delta_q;

    const ConservedVariablesGradType viscous_flux_m =
      viscous_terms.calculate_viscous_flux(u_m, grad_u_m);

    const ConservedVariablesGradType viscous_flux_p =
      viscous_terms.calculate_viscous_flux(u_p, grad_u_p);

    ConservedVariablesGradType total_flux_m =
      UtilityFunctions::dyadic_product(J_Rob, normal);
    total_flux_m += viscous_flux_p;
    total_flux_m *= alpha_1;
    total_flux_m += alpha_2 * viscous_flux_m;

    ConservedVariablesGradType total_flux_p =
      UtilityFunctions::dyadic_product(J_Rob, -normal); // opposite normal direction for phase 2
    total_flux_p += viscous_flux_m;
    total_flux_p *= alpha_2;
    total_flux_p += alpha_1 * viscous_flux_p;

    // penalty term

    const auto J_Dir_cons = calculate_Dirichlet_jump_in_conservative_variables<dim, Number, ConservedVariablesType, ConservedVariablesGradType>(u_m,
                                                       u_p,
                                                       specific_gas_constant_phase_1,
                                                       specific_gas_constant_phase_2,
                                                       gamma_m,
                                                       gamma_p,
                                                       m_dot_evap,
                                                       pressure_inf_phase_1,
                                                       pressure_inf_phase_2);

    // TODO: understand choice of alpha's
    const auto u_m_star = alpha_2 * u_m + alpha_1 * (u_p + J_Dir_cons);
    const auto u_p_star = alpha_1 * u_p + alpha_2 * (u_m - J_Dir_cons);

    ConservedVariablesGradType penalty_flux_m;
    const auto tmp_m = u_m - (u_p_star + J_Dir_cons);
    penalty_flux_m  = UtilityFunctions::dyadic_product(tmp_m, normal);
    penalty_flux_m *= tau;

    ConservedVariablesGradType penalty_flux_p;
    const auto tmp_p = u_p - (u_m_star - J_Dir_cons);
    penalty_flux_p  = UtilityFunctions::dyadic_product(tmp_p, -normal);
    penalty_flux_p *= tau;

    total_flux_m -= penalty_flux_m;
    total_flux_p -= penalty_flux_p;

    return std::make_pair(Flow::contract_tensor_with_normal(total_flux_m, normal),
                          Flow::contract_tensor_with_normal(total_flux_p, normal));
  }



  template <int dim, typename Number, typename ConservedVariablesType, typename ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesGradType, ConservedVariablesGradType>
    calculate_viscous_interface_flux_gradient(const ConservedVariablesType &u_m,
                                        const ConservedVariablesType &u_p,
                                        const Tensor<1,dim,VectorizedArray<Number>>              &normal,
                                        const double gamma_m,
                                        const double gamma_p,
                                        const double specific_gas_constant_phase_1,
                                        const double specific_gas_constant_phase_2,
                                        const double alpha_1,
                                        const double alpha_2,
                                        const double m_dot_evap,
                                        const double pressure_inf_phase_1,
                                        const double pressure_inf_phase_2,
                                        const MeltPoolDG::Flow::CompressibleFlowViscousKernels<dim, Number> &viscous_terms)
  {
    // TODO: add contributions for surface tension, interface heat source (laser energy) and Marangoni forces

    const auto J_Dir_cons = calculate_Dirichlet_jump_in_conservative_variables<dim, Number, ConservedVariablesType, ConservedVariablesGradType>(u_m,
                                                       u_p,
                                                       specific_gas_constant_phase_1,
                                                       specific_gas_constant_phase_2,
                                     				   gamma_m,
                                     				   gamma_p,
                                     				   m_dot_evap,
                                     				   pressure_inf_phase_1,
                                     				   pressure_inf_phase_2);

    const auto u_m_star = alpha_1 * u_m + alpha_2 * (u_p + J_Dir_cons);
    const auto u_p_star = alpha_2 * u_p + alpha_1 * (u_m - J_Dir_cons);

    auto tmp_m = u_m_star - u_m;
  	auto tmp_p = u_p_star - u_p;

    ConservedVariablesGradType arg_m =
      UtilityFunctions::dyadic_product(tmp_m, normal);

    ConservedVariablesGradType arg_p =
      UtilityFunctions::dyadic_product(tmp_p, -normal);

  	const ConservedVariablesGradType flux_grad_m =
    	viscous_terms.calculate_viscous_flux(u_m,
                           arg_m);

    const ConservedVariablesGradType flux_grad_p =
    	viscous_terms.calculate_viscous_flux(u_p,
                           arg_p);

    return std::make_pair(flux_grad_m, flux_grad_p);
  }
} // namespace MeltPoolDG::Multiphase
