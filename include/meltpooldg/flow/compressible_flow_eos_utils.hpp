/**
 * @brief A collection of helper functions, which depend on the equation of state.
 */

#pragma once

#include <meltpooldg/flow/compressible_flow_material_data.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>

namespace MeltPoolDG::Flow::EOS
{
  /**
   * Calculate the pressure from the conserved variables for a specific equation of state.
   *
   * @param conserved_variables Current values of the conserved variables.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Pressure resulting from the values of the conserved variables.
   */
  template <int dim, typename number = double, bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::VectorizedArray<number>
    calculate_thermodynamic_pressure(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowData<number>                               &flow_data)
  {
    const CompressibleFluidMaterialPhaseData<number> &material_data =
      is_gas_phase ? flow_data.material.gas : flow_data.material.liquid;

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
      MeltPoolDG::Flow::calculate_velocity<dim, number>(conserved_variables);

    switch (material_data.eos_data.type)
      {
          case EquationOfState::ideal_gas: {
            return (material_data.gamma - 1.) *
                   (conserved_variables[dim + 1] -
                    conserved_variables[0] * 0.5 * scalar_product(velocity, velocity));
          }
          case EquationOfState::stiffened_gas: {
            return (material_data.gamma - 1.) *
                     (conserved_variables[dim + 1] -
                      conserved_variables[0] * 0.5 * scalar_product(velocity, velocity)) -
                   material_data.gamma * material_data.eos_data.p_inf;
          }
          case EquationOfState::noble_abel_stiffened_gas: {
            return ((material_data.gamma - 1.) *
                      (conserved_variables[dim + 1] -
                       0.5 * conserved_variables[0] * scalar_product(velocity, velocity) -
                       conserved_variables[0] * material_data.eos_data.q) -
                    material_data.gamma * material_data.eos_data.p_inf *
                      (1. - conserved_variables[0] * material_data.eos_data.b)) /
                   (1. - conserved_variables[0] * material_data.eos_data.b);
          }
        default:
          AssertThrow(false, ExcNotImplemented("The given EOS is not supported."));
      }
  }

  /**
   * Calculate the gradient of the temperature from the conserved variables and their gradients for
   * a specific equation of state.
   *
   * @param conserved_variables Current values of the conserved variables.
   * @param grad_conserved_variables Current gradient of the conserved variables.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Current gradient of the temperature field.
   */
  template <int dim, typename number = double, bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    calculate_grad_T(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number>
                                         &grad_conserved_variables,
      const CompressibleFlowData<number> &flow_data)
  {
    const CompressibleFluidMaterialPhaseData<number> &material_data =
      is_gas_phase ? flow_data.material.gas : flow_data.material.liquid;

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> u =
      calculate_velocity<dim, number>(conserved_variables);
    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_u =
      calculate_grad_velocity<dim, number>(conserved_variables, grad_conserved_variables);
    const dealii::VectorizedArray<number> rho     = conserved_variables[0];
    const dealii::VectorizedArray<number> inv_rho = dealii::VectorizedArray<number>(1.) / rho;
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_rho =
      grad_conserved_variables[0];
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_E =
      inv_rho *
      (grad_conserved_variables[dim + 1] - inv_rho * conserved_variables[dim + 1] * grad_rho);

    switch (material_data.eos_data.type)
      {
          case EquationOfState::ideal_gas: {
            return (material_data.gamma - 1.0) / material_data.specific_gas_constant *
                   (grad_E - grad_u * u);
          }
        case EquationOfState::stiffened_gas:
          case EquationOfState::noble_abel_stiffened_gas: {
            return material_data.gamma / material_data.specific_isobaric_heat *
                   (grad_E - grad_u * u +
                    material_data.eos_data.p_inf * inv_rho * inv_rho * grad_rho);
          }
        default:
          AssertThrow(false, ExcNotImplemented("The given EOS is not supported."));
      }
  }

  /**
   * Calculate the speed of sound for a specific equation of state.
   *
   * @param conserved_variables Current values of the conserved variables.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Speed of sound resulting from the values of the conserved variables.
   */
  template <int dim, typename number = double, bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::VectorizedArray<number>
    calculate_speed_of_sound(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowData<number>                               &flow_data)
  {
    const CompressibleFluidMaterialPhaseData<number> &material_data =
      is_gas_phase ? flow_data.material.gas : flow_data.material.liquid;

    const auto pressure =
      calculate_thermodynamic_pressure<dim, number, is_gas_phase>(conserved_variables, flow_data);
    const auto density = conserved_variables[0];

    switch (material_data.eos_data.type)
      {
          case EquationOfState::ideal_gas: {
            return std::sqrt(material_data.gamma * pressure / density);
          }
          case EquationOfState::stiffened_gas: {
            return std::sqrt(material_data.gamma * (pressure + material_data.eos_data.p_inf) /
                             density);
          }
          case EquationOfState::noble_abel_stiffened_gas: {
            return std::sqrt(material_data.gamma * (pressure + material_data.eos_data.p_inf) /
                             (density * (1. - density * material_data.eos_data.b)));
          }
        default:
          AssertThrow(false, ExcNotImplemented("The given EOS is not supported."));
      }
  }

  /**
   * Calculate the temperature for a specific equation of state.
   *
   * @param conserved_variables Current values of the conserved variables.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Temperature resulting from the values of the conserved variables.
   */
  template <int dim, typename number = double, bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::VectorizedArray<number>
    calculate_temperature(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowData<number>                               &flow_data)
  {
    const CompressibleFluidMaterialPhaseData<number> &material_data =
      is_gas_phase ? flow_data.material.gas : flow_data.material.liquid;

    const auto pressure =
      calculate_thermodynamic_pressure<dim, number, is_gas_phase>(conserved_variables, flow_data);
    const auto density = conserved_variables[0];

    switch (material_data.eos_data.type)
      {
          case EquationOfState::ideal_gas: {
            return pressure / (material_data.specific_gas_constant * density);
          }
          case EquationOfState::stiffened_gas: {
            return (pressure + material_data.eos_data.p_inf) * material_data.gamma /
                   (material_data.specific_isobaric_heat * density * (material_data.gamma - 1.));
          }
          case EquationOfState::noble_abel_stiffened_gas: {
            return (pressure + material_data.eos_data.p_inf) *
                   (1. / density - material_data.eos_data.b) * material_data.gamma /
                   ((material_data.gamma - 1.) * material_data.specific_isobaric_heat);
          }
        default:
          AssertThrow(false, ExcNotImplemented("The given EOS is not supported."));
      }
  }

  /**
   * Calculate the total stress tensor from pressure contribution and viscous stress contribution.
   * sigma_ij = tau_ij - p * delta_ij
   *
   * @param conserved_variables Current values of the conserved variables.
   * @param grad_conserved_variables Current gradient of the conserved variables.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   * @param viscous_terms Collection of viscous term computations for the compressible Navier-Stokes
   * equations.
   *
   * @return Stress tensor resulting from the values and gradients of the conserved variables.
   */
  template <int dim, typename number = double, bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, VectorizedArray<number>>
    calculate_stress_tensor(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number>
                                         &grad_conserved_variables,
      const CompressibleFlowData<number> &flow_data,
      const auto                         &viscous_terms)
  {
    const auto grad_vel =
      Flow::calculate_grad_velocity<dim, number>(conserved_variables, grad_conserved_variables);

    const auto viscous_stress_tensor = viscous_terms.calculate_viscous_stress_tensor(grad_vel);

    const auto pressure_tensor =
      calculate_thermodynamic_pressure<dim, number, is_gas_phase>(conserved_variables, flow_data) *
      dealii::unit_symmetric_tensor<dim, VectorizedArray<number>>();

    return viscous_stress_tensor - pressure_tensor;
  }

  /**
   * Convert the given conservative variables (rho, momentum, total energy) to primitive variables
   * (pressure, velocity, temperature).
   *
   * @param u_cons Current values in conservative variables formulation.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Current values in primitive variables formulation.
   */
  template <int dim, typename number, bool is_gas_phase>
  inline DEAL_II_ALWAYS_INLINE //
    CompressibleFlowTypes::ConservedVariablesType<dim, number>
    convert_conservative_into_primitive_variables(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &u_cons,
      const Flow::CompressibleFlowData<number>                         &flow_data)
  {
    CompressibleFlowTypes::ConservedVariablesType<dim, number> u_prim;

    // pressure
    u_prim[0] = calculate_thermodynamic_pressure<dim, number, is_gas_phase>(u_cons, flow_data);

    // velocity
    for (unsigned int i = 1; i < dim + 1; i++)
      u_prim[i] = u_cons[i] / u_cons[0];

    // temperature
    u_prim[dim + 1] = calculate_temperature<dim, number, is_gas_phase>(u_cons, flow_data);

    return u_prim;
  }

  /**
   * Convert the given primitive variables (pressure, velocity, temperature) to conservative
   * variables (rho, momentum, total energy).
   *
   * @param u_prim Current values in primitive variables formulation.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Current values in conservative variables formulation.
   */
  template <int dim, typename number, bool is_gas_phase>
  inline DEAL_II_ALWAYS_INLINE //
    CompressibleFlowTypes::ConservedVariablesType<dim, number>
    convert_primitive_into_conservative_variables(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &u_prim,
      const Flow::CompressibleFlowData<number>                         &flow_data)
  {
    CompressibleFlowTypes::ConservedVariablesType<dim, number> u_cons;

    const CompressibleFluidMaterialPhaseData<number> &material_data =
      is_gas_phase ? flow_data.material.gas : flow_data.material.liquid;

    switch (material_data.eos_data.type)
      {
          case EquationOfState::ideal_gas: {
            // density
            u_cons[0] = u_prim[0] / (material_data.specific_gas_constant * u_prim[dim + 1]);
            // momentum
            for (unsigned int i = 1; i < dim + 1; i++)
              u_cons[i] = u_prim[i] * u_cons[0];
            // total energy
            const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
              MeltPoolDG::Flow::calculate_velocity<dim, number>(u_cons);
            u_cons[dim + 1] = u_cons[0] * (material_data.specific_gas_constant /
                                             (material_data.gamma - 1.) * u_prim[dim + 1] +
                                           0.5 * scalar_product(velocity, velocity));
            break;
          }
          case EquationOfState::stiffened_gas: {
            // density
            u_cons[0] =
              (u_prim[0] + material_data.eos_data.p_inf) * material_data.gamma /
              (material_data.specific_isobaric_heat * u_prim[dim + 1] * (material_data.gamma - 1.));
            // momentum
            for (unsigned int i = 1; i < dim + 1; i++)
              u_cons[i] = u_prim[i] * u_cons[0];
            // total energy
            const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
              MeltPoolDG::Flow::calculate_velocity<dim, number>(u_cons);
            u_cons[dim + 1] =
              u_cons[0] *
              (material_data.specific_isobaric_heat / material_data.gamma * u_prim[dim + 1] +
               material_data.eos_data.p_inf / u_cons[0] + 0.5 * scalar_product(velocity, velocity));
            break;
          }
          case EquationOfState::noble_abel_stiffened_gas: {
            // density
            u_cons[0] = (u_prim[0] + material_data.eos_data.p_inf) /
                        (material_data.specific_isobaric_heat * u_prim[dim + 1] *
                           (material_data.gamma - 1.) / material_data.gamma +
                         material_data.eos_data.b * (u_prim[0] + material_data.eos_data.p_inf));
            // momentum
            for (unsigned int i = 1; i < dim + 1; i++)
              u_cons[i] = u_prim[i] * u_cons[0];
            // total energy
            const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
              MeltPoolDG::Flow::calculate_velocity<dim, number>(u_cons);
            u_cons[dim + 1] =
              u_cons[0] *
              (material_data.specific_isobaric_heat / material_data.gamma * u_prim[dim + 1] +
               material_data.eos_data.p_inf * (1. / u_cons[0] - material_data.eos_data.b) +
               material_data.eos_data.q + 0.5 * scalar_product(velocity, velocity));
            break;
          }
        default:
          AssertThrow(false, ExcNotImplemented("The given EOS is not supported."));
      }

    return u_cons;
  }

  /**
   * Calculate the inner energy from a given pressure.
   *
   * @param pressure Given pressure value.
   * @param density Given density value.
   * @param material_data Collection of material parameters for the considered phase.
   *
   * @return Inner energy resulting from the given pressure and density values.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<number>
    compute_inner_energy_from_pressure(
      const VectorizedArray<number>                          &pressure,
      const VectorizedArray<number>                          &density,
      const Flow::CompressibleFluidMaterialPhaseData<number> &material_data)
  {
    switch (material_data.eos_data.type)
      {
          case EquationOfState::ideal_gas: {
            return pressure / (material_data.gamma - 1.);
          }
          case EquationOfState::stiffened_gas: {
            return (pressure + material_data.gamma * material_data.eos_data.p_inf) /
                   (material_data.gamma - 1.);
          }
          case EquationOfState::noble_abel_stiffened_gas: {
            return (pressure + material_data.gamma * material_data.eos_data.p_inf) /
                     (material_data.gamma - 1.) * (1. - density * material_data.eos_data.b) +
                   density * material_data.eos_data.q;
          }
        default:
          AssertThrow(false, ExcNotImplemented("The given EOS is not supported."));
      }
  }
} // namespace MeltPoolDG::Flow::EOS
