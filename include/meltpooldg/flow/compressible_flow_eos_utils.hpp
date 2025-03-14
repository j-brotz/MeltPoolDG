/**
* @brief A collection of helper functions for the evaluation of terms, which depend on the equation of state.
*/

#pragma once

#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_fluid_material_data.hpp>
#include <iomanip>

namespace MeltPoolDG::Flow
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
    calculate_pressure(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowData &flow_data)
    {
      const CompressibleFluidMaterialPhaseData<number> &material_data = is_gas_phase ? flow_data.material_data_gas_phase : flow_data.material_data_liquid_phase;

      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        MeltPoolDG::Flow::calculate_velocity<dim, number>(conserved_variables);

      switch (material_data.equation_of_state)
        {
          case EOS::ideal_gas: {
            return (material_data.gamma - 1.) * (conserved_variables[dim + 1] -
                             conserved_variables[0] * 0.5 * scalar_product(velocity, velocity));
          }
          case EOS::stiffened_gas: {
            return (material_data.gamma - 1.) * (conserved_variables[dim + 1] -
                             conserved_variables[0] * 0.5 * scalar_product(velocity, velocity))-material_data.gamma * material_data.eos_parameters.p_inf;
          }
          case EOS::noble_abel_stiffened_gas: {
            return ((material_data.gamma - 1.) * (conserved_variables[dim + 1]-
                             0.5 * conserved_variables[0] * scalar_product(velocity, velocity)-conserved_variables[0]*material_data.eos_parameters.q)-material_data.gamma * material_data.eos_parameters.p_inf * (1. - conserved_variables[0] * material_data.eos_parameters.b))/(1. - conserved_variables[0] * material_data.eos_parameters.b);
          }
          default:
           AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
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
      const CompressibleFlowData &flow_data)
    {
      const CompressibleFluidMaterialPhaseData<number> &material_data = is_gas_phase ? flow_data.material_data_gas_phase : flow_data.material_data_liquid_phase;

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

      switch (material_data.equation_of_state)
        {
          case EOS::ideal_gas: {
            return (material_data.gamma - 1.0) / material_data.specific_gas_constant * (grad_E - grad_u * u);
          }
          case EOS::stiffened_gas:
          case EOS::noble_abel_stiffened_gas: {
            return material_data.gamma / material_data.specific_isobaric_heat * (grad_E - grad_u * u + material_data.eos_parameters.p_inf * inv_rho * inv_rho * grad_rho);
          }
          default:
            AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
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
    const CompressibleFlowData &flow_data)
    {
      const CompressibleFluidMaterialPhaseData<number> &material_data = is_gas_phase ? flow_data.material_data_gas_phase : flow_data.material_data_liquid_phase;

      const auto pressure = calculate_pressure<dim, number, is_gas_phase>(conserved_variables, flow_data);
      const auto density = conserved_variables[0];

      switch (material_data.equation_of_state)
        {
          case EOS::ideal_gas: {
            return std::sqrt(material_data.gamma * pressure / density);
          }
          case EOS::stiffened_gas: {
            return std::sqrt(material_data.gamma * (pressure+material_data.eos_parameters.p_inf) / density);
          }
          case EOS::noble_abel_stiffened_gas: {
            return std::sqrt(material_data.gamma * (pressure+material_data.eos_parameters.p_inf) / (density*(1.-density*material_data.eos_parameters.b)));
          }
          default:
            AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
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
    const CompressibleFlowData &flow_data)
    {
      const CompressibleFluidMaterialPhaseData<number> &material_data = is_gas_phase ? flow_data.material_data_gas_phase : flow_data.material_data_liquid_phase;

      const auto pressure = calculate_pressure<dim, number, is_gas_phase>(conserved_variables, flow_data);
      const auto density = conserved_variables[0];

      switch (material_data.equation_of_state)
        {
          case EOS::ideal_gas: {
            return pressure / (material_data.specific_gas_constant * density);
          }
          case EOS::stiffened_gas: {
            return (pressure+material_data.eos_parameters.p_inf)*material_data.gamma/(material_data.specific_isobaric_heat*density*(material_data.gamma-1.));
          }
          case EOS::noble_abel_stiffened_gas: {
            return (pressure+material_data.eos_parameters.p_inf)*(1./density-material_data.eos_parameters.b)*material_data.gamma/((material_data.gamma-1.)*material_data.specific_isobaric_heat);
          }
          default:
            AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
        }
    }

  /**
   * Convert the given conservative variables (rho, momentum, total energy) to primitive variables
   * (pressure, velocity, temperature).
   *
   * @param u_cons Current values in conserved variable formulation.
   * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
   *
   * @return Current values in primitive variable formulation.
   */
  template <int dim, typename number, bool is_gas_phase>
  inline DEAL_II_ALWAYS_INLINE //
  CompressibleFlowTypes::ConservedVariablesType<dim, number>
  convert_conservative_into_primitive_variables(const CompressibleFlowTypes::ConservedVariablesType<dim, number>     &u_cons,
                                                const Flow::CompressibleFlowData &flow_data)
  {
    CompressibleFlowTypes::ConservedVariablesType<dim, number> u_prim;

    // pressure
    u_prim[0] = calculate_pressure<dim, number, is_gas_phase>(u_cons, flow_data);

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
  * @param u_prim Current values in primitive variable formulation.
  * @param flow_data Collection of parameters required by the compressible Navier-Stokes operator.
  *
  * @return Current values in conservative variable formulation.
  */
  template <int dim, typename number, bool is_gas_phase>
  inline DEAL_II_ALWAYS_INLINE //
  CompressibleFlowTypes::ConservedVariablesType<dim, number>
  convert_primitive_into_conservative_variables(const CompressibleFlowTypes::ConservedVariablesType<dim, number> &u_prim,
                                                const Flow::CompressibleFlowData &flow_data)
    {
      CompressibleFlowTypes::ConservedVariablesType<dim, number> u_cons;

      const CompressibleFluidMaterialPhaseData<number> &material_data = is_gas_phase ? flow_data.material_data_gas_phase : flow_data.material_data_liquid_phase;

      // momentum
          for (unsigned int i = 1; i < dim + 1; i++)
            u_cons[i] = u_prim[i] * u_cons[0];

      switch (material_data.equation_of_state)
        {
          case EOS::ideal_gas: {
            // density
            u_cons[0] = u_prim[0] / (material_data.specific_gas_constant * u_prim[dim + 1]);
            // total energy
            const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
              MeltPoolDG::Flow::calculate_velocity<dim, number>(u_cons);
            u_cons[dim + 1] = u_cons[0] * (material_data.specific_gas_constant / (material_data.gamma - 1.) *
              u_prim[dim + 1] + 0.5 * scalar_product(velocity,velocity));
            break;
          }
          case EOS::stiffened_gas: {
            // density
            u_cons[0] = (u_prim[0] + material_data.eos_parameters.p_inf) * material_data.gamma / (material_data.specific_isobaric_heat * u_prim[dim + 1] * (material_data.gamma-1.));
            // total energy
            const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
              MeltPoolDG::Flow::calculate_velocity<dim, number>(u_cons);
            u_cons[dim + 1] = u_cons[0] * (material_data.specific_isobaric_heat/material_data.gamma *
              u_prim[dim + 1] + material_data.eos_parameters.p_inf/u_cons[0] + 0.5 * scalar_product(velocity,velocity));
            break;
          }
          case EOS::noble_abel_stiffened_gas: {
            // density
            u_cons[0] = (u_prim[0]+material_data.eos_parameters.p_inf) / (material_data.specific_isobaric_heat * u_prim[dim + 1] * (material_data.gamma-1.)/material_data.gamma + material_data.eos_parameters.b *(u_prim[0]+material_data.eos_parameters.p_inf));
            // total energy
            const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
            MeltPoolDG::Flow::calculate_velocity<dim, number>(u_cons);
            u_cons[dim + 1] = u_cons[0] * (material_data.specific_isobaric_heat/material_data.gamma *
              u_prim[dim + 1] + material_data.eos_parameters.p_inf * (1./u_cons[0] -material_data.eos_parameters.b)+material_data.eos_parameters.q + 0.5 * scalar_product(velocity,velocity));
            break;
          }
          default:
            AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
        }

      return u_cons;
    }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<number>
  compute_inner_energy_from_pressure(const VectorizedArray<number> &pressure,
                                      const VectorizedArray<number> &density,
                                      const Flow::CompressibleFluidMaterialPhaseData<number> &material_data)
    {
      switch (material_data.equation_of_state)
        {
          case EOS::ideal_gas: {
            return pressure / (material_data.gamma - 1.);
            break;
          }
          case EOS::stiffened_gas: {
            return (pressure+material_data.gamma * material_data.eos_parameters.p_inf) / (material_data.gamma - 1.);
            break;
          }
          case EOS::noble_abel_stiffened_gas: {
            return (pressure+material_data.gamma * material_data.eos_parameters.p_inf) / (material_data.gamma - 1.)
                            * (1.-density * material_data.eos_parameters.b) + density * material_data.eos_parameters.q;
            break;
          }
          default:
            AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
        }
    }
} // namespace MeltPoolDG::Flow