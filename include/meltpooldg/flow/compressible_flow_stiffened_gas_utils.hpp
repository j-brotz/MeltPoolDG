#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/compressible_flow_eos_utils_base.hpp>
#include <meltpooldg/flow/compressible_flow_material_data.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>

namespace MeltPoolDG::Flow::EOS
{
  /**
   * @brief A class for a collection of thermodynamic helper functions for the stiffened gas
   * equation of state.
   */
  template <int dim, typename number>
  class StiffenedGas : public EquationOfStateUtils<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param material_data_in Reference to a material data object providing all relevant material
     * parameters.
     */
    explicit StiffenedGas(const CompressibleFluidMaterialPhaseData<number> &material_data_in)
      : material_data(material_data_in)
    {}

    /**
     * @brief Calculate the pressure from the conserved variables for a stiffened gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Pressure resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_thermodynamic_pressure(const CompressibleFlow::ConservedVariablesType<dim, number>
                                         &conserved_variables) const override
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(conserved_variables);

      return (material_data.gamma - 1.) *
               (conserved_variables[dim + 1] -
                conserved_variables[0] * 0.5 * scalar_product(velocity, velocity)) -
             material_data.gamma * material_data.eos_data.p_inf;
    }

    /**
     * @brief Calculate the gradient of the temperature from the conserved variables and their
     * gradients for a stiffened gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the temperature field.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
      calculate_grad_T(
        const CompressibleFlow::ConservedVariablesType<dim, number> &conserved_variables,
        const CompressibleFlow::ConservedVariablesGradientType<dim, number>
          &grad_conserved_variables) const override
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

      return material_data.gamma / material_data.specific_isobaric_heat *
             (grad_E - grad_u * u + material_data.eos_data.p_inf * inv_rho * inv_rho * grad_rho);
    }

    /**
     * @brief Calculate the speed of sound for a stiffened gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Speed of sound resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_speed_of_sound(const CompressibleFlow::ConservedVariablesType<dim, number>
                                 &conserved_variables) const override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return std::sqrt(material_data.gamma * (pressure + material_data.eos_data.p_inf) / density);
    }

    /**
     * @brief Calculate the temperature for a stiffened gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Temperature resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_temperature(const CompressibleFlow::ConservedVariablesType<dim, number>
                              &conserved_variables) const override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return (pressure + material_data.eos_data.p_inf) * material_data.gamma /
             (material_data.specific_isobaric_heat * density * (material_data.gamma - 1.));
    }

    /**
     * @brief Convert the given primitive variables (pressure, velocity, temperature) to conservative
     * variables (rho, momentum, total energy).
     *
     * @param u_prim Current values in primitive variables formulation.
     *
     * @return Current values in conservative variables formulation.
     */
    inline DEAL_II_ALWAYS_INLINE //
      CompressibleFlow::ConservedVariablesType<dim, number>
      convert_primitive_into_conservative_variables(
        const CompressibleFlow::ConservedVariablesType<dim, number> &u_prim) const override
    {
      CompressibleFlow::ConservedVariablesType<dim, number> u_cons;

      // density
      u_cons[0] =
        (u_prim[0] + material_data.eos_data.p_inf) * material_data.gamma /
        (material_data.specific_isobaric_heat * u_prim[dim + 1] * (material_data.gamma - 1.));
      // momentum
      for (unsigned int i = 1; i < dim + 1; i++)
        u_cons[i] = u_prim[i] * u_cons[0];
      // total energy
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(u_cons);
      u_cons[dim + 1] =
        u_cons[0] *
        (material_data.specific_isobaric_heat / material_data.gamma * u_prim[dim + 1] +
         material_data.eos_data.p_inf / u_cons[0] + 0.5 * scalar_product(velocity, velocity));

      return u_cons;
    }

    /**
     * @brief Calculate the inner energy from a given pressure.
     *
     * @param pressure Given pressure value.
     *
     * @return Inner energy resulting from the given pressure and density values.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      compute_inner_energy_from_pressure(const dealii::VectorizedArray<number> &pressure,
                                         const dealii::VectorizedArray<number> &) const override
    {
      return (pressure + material_data.gamma * material_data.eos_data.p_inf) /
             (material_data.gamma - 1.);
    }

  private:
    /// Material data object providing all relevant material parameters
    const CompressibleFluidMaterialPhaseData<number> &material_data;
  };
} // namespace MeltPoolDG::Flow::EOS
