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
      const CompressibleFluidMaterialPhaseData<number> &material_data = is_gas_phase ? flow_data.material_data_gas_phase : flow_data.material_data_liquid_phase;

      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        MeltPoolDG::Flow::calculate_velocity<dim, number>(conserved_variables);

      switch (flow_data.material_data_gas_phase.equation_of_state)
        {
          case EOS::ideal_gas: {
            return (material_data.gamma - 1.) * (conserved_variables[dim + 1] -
                             conserved_variables[0] * 0.5 * scalar_product(velocity, velocity));
          }
          case EOS::stiffened_gas: {
            return (material_data.gamma - 1.) * (conserved_variables[dim + 1] -
                             conserved_variables[0] * 0.5 * scalar_product(velocity, velocity))-material_data.gamma * material_data.eos_parameters.p_inf;
          }
          case EOS::noble_abel_stiffend_gas: {
            return (material_data.gamma - 1.) * (conserved_variables[dim + 1] -
                             conserved_variables[0] * 0.5 * scalar_product(velocity, velocity)-material_data.eos_parameters.q)*(material_data.gamma-1.)/(1./conserved_variables[0] * material_data.eos_parameters.b)-material_data.gamma * material_data.eos_parameters.p_inf;
          }
          default:
           AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
        }
    }

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

      switch (flow_data.material_data_gas_phase.equation_of_state)
        {
          case EOS::ideal_gas: {
            return (material_data.gamma - 1.0) / material_data.specific_gas_constant * (grad_E - grad_u * u);
          }
          case EOS::stiffened_gas:
          case EOS::noble_abel_stiffend_gas: {
            return material_data.gamma / material_data.specific_isobaric_heat * (grad_E - grad_u * u + material_data.eos_parameters.p_inf * inv_rho * inv_rho * grad_rho);
          }
          default:
            AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
        }
    }

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

      switch (flow_data.material_data_gas_phase.equation_of_state)
        {
          case EOS::ideal_gas: {
            return std::sqrt(material_data.gamma * pressure / density);
          }
          case EOS::stiffened_gas: {
            return std::sqrt(material_data.gamma * (pressure+material_data.eos_parameters.p_inf) / density);
          }
          case EOS::noble_abel_stiffend_gas: {
            return std::sqrt(material_data.gamma * (pressure+material_data.eos_parameters.p_inf) / (density*(1.-density*material_data.eos_parameters.b)));
          }
          default:
            AssertThrow(false, ExcNotImplemented("This code section should never be reached!"));
        }
    }
} // namespace MeltPoolDG::Flow