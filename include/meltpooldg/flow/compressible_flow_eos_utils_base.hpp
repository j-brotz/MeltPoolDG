#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>

namespace MeltPoolDG::Flow::EOS
{
  /**
   * @brief A base class for a collection of thermodynamic helper functions, which depend on the
   * equation of state.
   */
  template <int dim, typename number>
  class EquationOfStateUtils
  {
  public:
    virtual ~EquationOfStateUtils() = default;

    /**
     * @brief Calculate the pressure from the conserved variables for a specific equation of state.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Pressure resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::VectorizedArray<number>
      calculate_thermodynamic_pressure(
        const CompressibleFlow::ConservedVariablesType<dim, number> &conserved_variables) const = 0;

    /**
     * @brief Calculate the gradient of the temperature from the conserved variables and their gradients
     * for a specific equation of state.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the temperature field.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
      calculate_grad_T(
        const CompressibleFlow::ConservedVariablesType<dim, number> &conserved_variables,
        const CompressibleFlow::ConservedVariablesGradientType<dim, number>
          &grad_conserved_variables) const = 0;

    /**
     * @brief Calculate the speed of sound for a specific equation of state.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Speed of sound resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::VectorizedArray<number>
      calculate_speed_of_sound(
        const CompressibleFlow::ConservedVariablesType<dim, number> &conserved_variables) const = 0;

    /**
     * @brief Calculate the temperature for a specific equation of state.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Temperature resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::VectorizedArray<number>
      calculate_temperature(
        const CompressibleFlow::ConservedVariablesType<dim, number> &conserved_variables) const = 0;

    /**
     * @brief Calculate the total stress tensor from pressure contribution and viscous stress
     * contribution sigma_ij = tau_ij - p * delta_ij.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param viscous_stress_tensor Given viscous tress tensor.
     *
     * @return Stress tensor resulting from the values and gradients of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
      calculate_stress_tensor(
        const CompressibleFlow::ConservedVariablesType<dim, number>   &conserved_variables,
        const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &viscous_stress_tensor) const
    {
      const auto pressure_tensor =
        calculate_thermodynamic_pressure(conserved_variables) *
        dealii::unit_symmetric_tensor<dim, dealii::VectorizedArray<number>>();

      return viscous_stress_tensor - pressure_tensor;
    }

    /**
     * @brief Convert the given conservative variables (rho, momentum, total energy) to primitive
     * variables (pressure, velocity, temperature).
     *
     * @param u_cons Current values in conservative variables formulation.
     *
     * @return Current values in primitive variables formulation.
     */
    inline DEAL_II_ALWAYS_INLINE //
      CompressibleFlow::ConservedVariablesType<dim, number>
      convert_conservative_into_primitive_variables(
        const CompressibleFlow::ConservedVariablesType<dim, number> &u_cons) const
    {
      CompressibleFlow::ConservedVariablesType<dim, number> u_prim;

      // pressure
      u_prim[0] = calculate_thermodynamic_pressure(u_cons);

      // velocity
      for (unsigned int i = 1; i < dim + 1; i++)
        u_prim[i] = u_cons[i] / u_cons[0];

      // temperature
      u_prim[dim + 1] = calculate_temperature(u_cons);

      return u_prim;
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
      virtual CompressibleFlow::ConservedVariablesType<dim, number>
      convert_primitive_into_conservative_variables(
        const CompressibleFlow::ConservedVariablesType<dim, number> &u_prim) const = 0;

    /**
     * @brief Calculate the inner energy from a given pressure.
     *
     * @param pressure Given pressure value.
     * @param density Given density value.
     *
     * @return Inner energy resulting from the given pressure and density values.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::VectorizedArray<number>
      compute_inner_energy_from_pressure(const dealii::VectorizedArray<number> &pressure,
                                         const dealii::VectorizedArray<number> &density) const = 0;
  };
} // namespace MeltPoolDG::Flow::EOS
