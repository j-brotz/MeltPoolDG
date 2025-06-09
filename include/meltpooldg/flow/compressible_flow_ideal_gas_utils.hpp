/**
 * @brief A class for a collection of thermodynamic helper functions for the ideal gas equation of
 * state.
 */

#pragma once

#include <meltpooldg/flow/compressible_flow_material_data.hpp>

namespace MeltPoolDG::Flow::EOS
{
  template <int dim, typename number>
  class IdealGas : public EquationOfStateUtils<dim, number>
  {
  public:
    /**
     * Constructor.
     *
     * @param material_data_in Reference to a material data object providing all relevant material
     * parameters.
     */
    explicit IdealGas(const CompressibleFluidMaterialPhaseData<number> &material_data_in)
      : material_data(material_data_in)
    {}

    /**
     * Calculate the pressure from the conserved variables for an ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Pressure resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_thermodynamic_pressure(
        const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables)
        override
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(conserved_variables);

      return (material_data.gamma - 1.) *
             (conserved_variables[dim + 1] -
              conserved_variables[0] * 0.5 * scalar_product(velocity, velocity));
    }

    /**
     * Calculate the gradient of the temperature from the conserved variables and their gradients
     * for an ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the temperature field.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
      calculate_grad_T(
        const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
        const CompressibleFlowTypes::ConservedVariablesGradType<dim, number>
          &grad_conserved_variables) override
    {
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

      return (material_data.gamma - 1.0) / material_data.specific_gas_constant *
             (grad_E - grad_u * u);
    }

    /**
     * Calculate the speed of sound for an ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Speed of sound resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_speed_of_sound(const CompressibleFlowTypes::ConservedVariablesType<dim, number>
                                 &conserved_variables) override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return std::sqrt(material_data.gamma * pressure / density);
    }

    /**
     * Calculate the temperature for an ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Temperature resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_temperature(const CompressibleFlowTypes::ConservedVariablesType<dim, number>
                              &conserved_variables) override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return pressure / (material_data.specific_gas_constant * density);
    }

    /**
     * Convert the given primitive variables (pressure, velocity, temperature) to conservative
     * variables (rho, momentum, total energy).
     *
     * @param u_prim Current values in primitive variables formulation.
     *
     * @return Current values in conservative variables formulation.
     */
    inline DEAL_II_ALWAYS_INLINE //
      CompressibleFlowTypes::ConservedVariablesType<dim, number>
      convert_primitive_into_conservative_variables(
        const CompressibleFlowTypes::ConservedVariablesType<dim, number> &u_prim) override
    {
      CompressibleFlowTypes::ConservedVariablesType<dim, number> u_cons;

      // density
      u_cons[0] = u_prim[0] / (material_data.specific_gas_constant * u_prim[dim + 1]);
      // momentum
      for (unsigned int i = 1; i < dim + 1; i++)
        u_cons[i] = u_prim[i] * u_cons[0];
      // total energy
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(u_cons);
      u_cons[dim + 1] = u_cons[0] * (material_data.specific_gas_constant /
                                       (material_data.gamma - 1.) * u_prim[dim + 1] +
                                     0.5 * scalar_product(velocity, velocity));
      return u_cons;
    }

    /**
     * Calculate the inner energy from a given pressure.
     *
     * @param pressure Given pressure value.
     *
     * @return Inner energy resulting from the given pressure and density values.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      compute_inner_energy_from_pressure(const dealii::VectorizedArray<number> &pressure,
                                         const dealii::VectorizedArray<number> &) override
    {
      return pressure / (material_data.gamma - 1.);
    }

  private:
    /// material data object providing all relevant material parameters
    const CompressibleFluidMaterialPhaseData<number> &material_data;
  };
} // namespace MeltPoolDG::Flow::EOS
