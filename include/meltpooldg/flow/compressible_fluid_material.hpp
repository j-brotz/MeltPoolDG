#pragma once

#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_fluid_material_data.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * ...
   */
    template <int dim, typename number = double, bool is_gas_phase = true>
    inline DEAL_II_ALWAYS_INLINE //
    dealii::VectorizedArray<number>
    calculate_pressure(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowData &flow_data)
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        MeltPoolDG::Flow::calculate_velocity<dim, number>(conserved_variables);

      number gamma;
      if (is_gas_phase)
        gamma = flow_data.gamma;
      else
        gamma = flow_data.gamma_2;

      switch (flow_data.equation_of_state)
        {
          case EOS::ideal_gas: {
            return (gamma - 1.) * (conserved_variables[dim + 1] -
                             conserved_variables[0] * 0.5 * scalar_product(velocity, velocity));
          }
          case EOS::stiffened_gas: {
            number p_inf;
            if (is_gas_phase)
              p_inf = flow_data.p_inf;
            else
              p_inf = flow_data.p_inf_2;
            return (gamma - 1.) * (conserved_variables[dim + 1] -
                             conserved_variables[0] * 0.5 * scalar_product(velocity, velocity))-gamma * p_inf;
          }
          case EOS::noble_abel_stiffend_gas: {
            number p_inf;
            if (is_gas_phase)
              p_inf = flow_data.p_inf;
            else
              p_inf = flow_data.p_inf_2;
            number b;
            if (is_gas_phase)
              b = flow_data.b;
            else
              b = flow_data.b_2;
            number q;
            if (is_gas_phase)
              q = flow_data.q;
            else
              q = flow_data.q_2;
            return (gamma - 1.) * (conserved_variables[dim + 1] -
                             conserved_variables[0] * 0.5 * scalar_product(velocity, velocity)-q)*(gamma-1.)/(1./conserved_variables[0] * b)-gamma * p_inf;
          }
          default:
           AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
        }
    }

    /*template <int dim, typename number = double, bool is_gas_phase = true>
    inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    calculate_grad_T(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number>
      &grad_conserved_variables,
      const CompressibleFlowData &flow_data)
    {
      const number gamma = is_gas_phase ?flow_data.gamma
                                    : flow_data.gamma_2;
      const number specific_gas_constant = is_gas_phase ? flow_data.specific_gas_constant
                                    : flow_data.specific_gas_constant_2;

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
      return (gamma - 1.0) / specific_gas_constant * (grad_E - grad_u * u);
    }

    template <int dim, typename number = double, bool is_gas_phase = true>
    inline DEAL_II_ALWAYS_INLINE //
    dealii::VectorizedArray<number>
    calculate_speed_of_sound(
    const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
    const CompressibleFlowData &flow_data)
    {
      const auto pressure = calculate_pressure(conserved_variables);

      const auto speed_of_sound =
        std::sqrt(flow_data.gamma * pressure / conserved_variables[0]);

      return speed_of_sound;
    }*/
} // namespace MeltPoolDG::Flow