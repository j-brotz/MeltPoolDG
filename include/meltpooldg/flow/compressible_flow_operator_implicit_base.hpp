#pragma once

#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/flow/compressible_flow_operator_base.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  class CompressibleFlowOperatorImplicitBase : public CompressibleFlowOperatorBase<dim, number>
  {
  private:
    using ConservedVariablesType =
      typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesType;
    using ConservedVariablesGradType =
      typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesGradType;
    using VectorType = typename CompressibleFlowOperatorBase<dim, number>::VectorType;

  public:
    /**
     * Cosntructor, calls the base class constructor and precomputes all required constant
     * variables.
     *
     * @param compressible_flow_data_in Reference to the compressible flow data struct used.
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param solution_history_in Reference to the used solution_history object.
     * @param comp_flow_dof_idx_in Index of the used dof handler in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     */
    CompressibleFlowOperatorImplicitBase(
      const CompressibleFlowData                     &compressible_flow_data_in,
      const ScratchData<dim>                         &scratch_data_in,
      ::TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
      unsigned int                                    comp_flow_dof_idx_in  = 0,
      unsigned int                                    comp_flow_quad_idx_in = 0);

  protected:
    // TODO: Check performance impact when the functions below are inlined.
    /**
     * Compute the linearization of the convective numerical flux with respect to the primary
     * variables.
     *
     * @param w_q Primary variables on the inner (first) and outer (second) face.
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param normal Outer facing normal vector.
     * @return Linearized convective numerical flux.
     */
    ConservedVariablesGradType
    calculate_jacobian_convective_numerical_flux(
      const std::pair<ConservedVariablesType, ConservedVariablesType> &w_q,
      const std::pair<ConservedVariablesType, ConservedVariablesType> &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>   &normal) const;

    /**
     * Compute the linearization of the viscous numerical flux with respect to the primary
     * variables.
     *
     * @param w_q Primary variables on the inner (first) and outer (second) face.
     * @param grad_w_q Gradient of the primary variables on the inner (first) and outer
     * (second) face.
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param grad_delta_w_q Gradient of the change in the primary variables on inner (first) and
     * outer (second) face.
     * @param normal Outer facing normal vector.
     * @param penalty_parameter Value of the symmetric interior penalty parameter.
     * @return Linearized viscous numerical flux.
     */
    ConservedVariablesGradType
    calculate_jacobian_viscous_numerical_flux(
      const std::pair<ConservedVariablesType, ConservedVariablesType>         &w_q,
      const std::pair<ConservedVariablesGradType, ConservedVariablesGradType> &grad_w_q,
      const std::pair<ConservedVariablesType, ConservedVariablesType>         &delta_w_q,
      const std::pair<ConservedVariablesGradType, ConservedVariablesGradType> &grad_delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>           &normal,
      dealii::VectorizedArray<number> penalty_parameter) const;

    /**
     * Compute the linearization of the convective flux with respect to the primary variables.
     *
     * @param w_q Primary variables.
     * @param delta_w_q Change in the primary variables.
     * @return Linearized convective flux.
     */
    ConservedVariablesGradType
    calculate_jacobain_convective_flux(const ConservedVariablesType &w_q,
                                       const ConservedVariablesType &delta_w_q) const;

    /**
     * Compute the linearization of the viscous flux with respect to the primary variables.
     *
     * @param w_q Primary variables.
     * @param grad_w_q Gradient of the primary variables.
     * @param delta_w_q Change in the primary variables.
     * @param grad_delta_w_q Gradient of the change in the primary variables.
     * @return Linearized viscous flux.
     */
    ConservedVariablesGradType
    calculate_jacobian_viscous_flux(const ConservedVariablesType     &w_q,
                                    const ConservedVariablesGradType &grad_w_q,
                                    const ConservedVariablesType     &delta_w_q,
                                    const ConservedVariablesGradType &grad_delta_w_q) const;

    /**
     * Compute the jump term in the viscous numerical flux. For the used symmetric interior penalty
     * approach the jump therm is given by the penalty parameter multiplied with the jump in the
     * primary variables.
     *
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param normal Outer facing normal vector.
     * @param penalty_parameter Value of the symmetric interior penalty parameter.
     * @return Jump term of the symmetric interior penalty flux.
     */
    ConservedVariablesGradType
    calculate_jacobian_viscous_numerical_flux_jump_term(
      const std::pair<ConservedVariablesType, ConservedVariablesType> &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>   &normal,
      dealii::VectorizedArray<number>                                  penalty_parameter) const;

    /**
     * Compute the jump term in the convective numerical flux. For the used Lax-Friedrichs flux the
     * jump term is given by lambda/2 times the jump in the primary variables.
     *
     * @param w_q Primary variables on the inner (first) and outer (second) face.
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param normal Outer facing normal vector.
     * @return Jump term of the Lax-Friedrichs flux.
     */
    ConservedVariablesGradType
    calculate_jacobian_convective_numerical_flux_jump_term(
      const std::pair<ConservedVariablesType, ConservedVariablesType> &w_q,
      const std::pair<ConservedVariablesType, ConservedVariablesType> &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>   &normal) const;

    /**
     * This function sets the corresponding values on the fictional outer face if the face is
     * located at a boundary when computing the jacobian.
     *
     * @param q_point Coordinates of the quadratiure point.
     * @param normal Outer facing normal vector.
     * @param boundary_id ID of the current boundary.
     * @param w_m Primary variables on the inner face.
     * @param delta_w_m Change in the primary variables on the inner face.
     * @param grad_w_m Gradient of the primary variables on the inner face.
     * @param grad_delta_w_m Gradient of the change in the primary variables on the inner face.
     *
     * @return Tuple containing the corresponding values on the outer face. The first value being
     * the primary variables, the second the gradient of the primary variables, the third the change
     * in the primary variables and the fourth the change in the primary variables.
     */
    std::tuple<ConservedVariablesType,
               ConservedVariablesGradType,
               ConservedVariablesType,
               ConservedVariablesGradType>
    get_adjacent_jacobian_face_values_at_boundary(
      const Point<dim, VectorizedArray<number>>     &q_point,
      const Tensor<1, dim, VectorizedArray<number>> &normal,
      unsigned int                                   boundary_id,
      const ConservedVariablesType                  &w_m,
      const ConservedVariablesType                  &delta_w_m,
      const ConservedVariablesGradType              &grad_w_m,
      const ConservedVariablesGradType              &grad_delta_w_m) const;

    // precomputed constants
    number rs_div_c;
    number lambda_div_c;
  };


  /**
   * Inline functions
   */
  template <int dim, typename number>
  typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesGradType
  CompressibleFlowOperatorImplicitBase<dim, number>::calculate_jacobian_convective_numerical_flux(
    const std::pair<ConservedVariablesType, ConservedVariablesType> &w_q,
    const std::pair<ConservedVariablesType, ConservedVariablesType> &delta_w_q,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>   &normal) const
  {
    // For now only the exact and modified Lax-Friedrichs flux are supported
    AssertThrow(this->comp_flow_data.numerical_flux_type == "lax_friedrichs_exact" ||
                  this->comp_flow_data.numerical_flux_type == "lax_friedrichs_modified",
                dealii::ExcMessage("The chosen convective numerical flux type '" +
                                   this->comp_flow_data.numerical_flux_type +
                                   "' is not supported within the analytic Jacobian computation."));

    // average
    ConservedVariablesGradType flux_p =
      calculate_jacobain_convective_flux(w_q.second, delta_w_q.second);
    ConservedVariablesGradType flux_m =
      calculate_jacobain_convective_flux(w_q.first, delta_w_q.first);

    ConservedVariablesGradType flux;
    for (unsigned int i = 0; i < dim + 2; ++i)
      flux[i] = 0.5 * (flux_p[i] + flux_m[i]);

    flux += calculate_jacobian_convective_numerical_flux_jump_term(w_q, delta_w_q, normal);

    return flux;
  }

  template <int dim, typename number>
  typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesGradType
  CompressibleFlowOperatorImplicitBase<dim, number>::calculate_jacobian_viscous_numerical_flux(
    const std::pair<ConservedVariablesType, ConservedVariablesType>         &w_q,
    const std::pair<ConservedVariablesGradType, ConservedVariablesGradType> &grad_w_q,
    const std::pair<ConservedVariablesType, ConservedVariablesType>         &delta_w_q,
    const std::pair<ConservedVariablesGradType, ConservedVariablesGradType> &grad_delta_w_q,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>           &normal,
    dealii::VectorizedArray<number> penalty_parameter) const
  {
    ConservedVariablesGradType flux_p =
      0.5 * calculate_jacobian_viscous_flux(w_q.second,
                                            grad_w_q.second,
                                            delta_w_q.second,
                                            grad_delta_w_q.second);
    ConservedVariablesGradType flux_m = 0.5 * calculate_jacobian_viscous_flux(w_q.first,
                                                                              grad_w_q.first,
                                                                              delta_w_q.first,
                                                                              grad_delta_w_q.first);

    ConservedVariablesGradType flux;
    for (unsigned int i = 0; i < dim + 2; ++i)
      flux[i] = (flux_p[i] + flux_m[i]);

    flux -=
      calculate_jacobian_viscous_numerical_flux_jump_term(delta_w_q, normal, penalty_parameter);
    return flux;
  }

  template <int dim, typename number>
  typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesGradType
  CompressibleFlowOperatorImplicitBase<dim, number>::calculate_jacobain_convective_flux(
    const ConservedVariablesType &w_q,
    const ConservedVariablesType &delta_w_q) const
  {
    ConservedVariablesGradType convective_differential_change;

    // precompute values
    dealii::VectorizedArray<number> rho_inv                               = 1.0 / w_q[0];
    dealii::VectorizedArray<number> momentum_norm_squared                 = 0.0;
    dealii::VectorizedArray<number> momentum_times_delta_momentum_squared = 0.0;
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum;
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_momentum;
    for (unsigned int i = 1; i < dim + 1; ++i)
      {
        momentum_norm_squared += w_q[i] * w_q[i];
        momentum_times_delta_momentum_squared += delta_w_q[i] * w_q[i];
        momentum[i - 1]       = w_q[i];
        delta_momentum[i - 1] = delta_w_q[i];
      }

    //** change in density flux **//
    for (unsigned int dimension = 0; dimension < dim; ++dimension)
      convective_differential_change[0][dimension] = delta_w_q[dimension + 1];

    //** change in momentum flux **//
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> helper;
    // density direction
    helper -=
      rho_inv * rho_inv * delta_w_q[0] *
      VectorTools::dyadic_product<dim, dim, dealii::VectorizedArray<number>>(w_q.begin_raw() + 1,
                                                                             w_q.begin_raw() + 1);
    helper += rho_inv * rho_inv * rs_div_c * momentum_norm_squared * delta_w_q[0] * 0.5 *
              VectorTools::identity<dim, dealii::VectorizedArray<number>>();
    // momentum direction
    helper += rho_inv * VectorTools::dyadic_product<dim, dim, dealii::VectorizedArray<number>>(
                          delta_w_q.begin_raw() + 1, w_q.begin_raw() + 1);
    helper += rho_inv * VectorTools::dyadic_product<dim, dim, dealii::VectorizedArray<number>>(
                          w_q.begin_raw() + 1, delta_w_q.begin_raw() + 1);
    helper -= rs_div_c * rho_inv * momentum_times_delta_momentum_squared *
              VectorTools::identity<dim, dealii::VectorizedArray<number>>();
    // energy direction
    helper +=
      rs_div_c * delta_w_q[dim + 1] * VectorTools::identity<dim, dealii::VectorizedArray<number>>();

    for (unsigned int i = 0; i < dim; ++i)
      {
        convective_differential_change[i + 1] += helper[i];
      }

    //** change in energy density flux **//
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> helper_energy;
    // density part
    helper_energy -= (w_q[dim + 1] + rs_div_c * (w_q[dim + 1] - rho_inv * momentum_norm_squared)) *
                     rho_inv * rho_inv * delta_w_q[0] * momentum;
    // momentum part
    helper_energy +=
      (w_q[dim + 1] + rs_div_c * (w_q[dim + 1] - 0.5 * rho_inv * momentum_norm_squared)) * rho_inv *
      delta_momentum;
    helper_energy -=
      rs_div_c * momentum_times_delta_momentum_squared * rho_inv * rho_inv * momentum;
    // energy part
    helper_energy += (1 + rs_div_c) * delta_w_q[dim + 1] * rho_inv * momentum;
    convective_differential_change[dim + 1] = helper_energy;

    return convective_differential_change;
  }

  template <int dim, typename number>
  typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesGradType
  CompressibleFlowOperatorImplicitBase<dim, number>::calculate_jacobian_viscous_flux(
    const ConservedVariablesType     &w_q,
    const ConservedVariablesGradType &grad_w_q,
    const ConservedVariablesType     &delta_w_q,
    const ConservedVariablesGradType &grad_delta_w_q) const
  {
    ConservedVariablesGradType viscous_differential_change;

    // precompute values
    dealii::VectorizedArray<number>                         rho_inv = 1.0 / w_q[0];
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> m_q;
    for (unsigned int i = 0; i < dim; ++i)
      m_q[i] = w_q[i + 1];
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      delta_m_q[i] = delta_w_q[i + 1];

    /* change in density flux */
    viscous_differential_change[0] = dealii::Tensor<1, dim, dealii::VectorizedArray<number>>();

    /* change in momentum flux */
    // density part
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      grad_m_q[i] = grad_w_q[i + 1];
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_delta_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      grad_delta_m_q[i] = grad_delta_w_q[i + 1];

    auto       v_q           = this->calculator_functions.calculate_velocity(w_q);
    const auto grad_v_q_temp = this->calculator_functions.calculate_grad_velocity(w_q, grad_w_q);
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_v_q;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        grad_v_q[i][j] = grad_v_q_temp[i][j];

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_a =
      rho_inv * (-delta_w_q[0] * grad_v_q);
    param_a += rho_inv * rho_inv * delta_w_q[0] * VectorTools::dyadic_product(v_q, grad_w_q[0]);
    param_a -= rho_inv * VectorTools::dyadic_product(v_q, grad_delta_w_q[0]);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_c =
      param_a;
    param_c += VectorTools::transpose(param_a);
    param_c -= 2. / 3. * VectorTools::trace(param_a) *
               VectorTools::identity<dim, dealii::VectorizedArray<number>>();
    param_c *= this->comp_flow_data.dynamic_viscosity;

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_b =
      rho_inv * grad_delta_m_q;
    param_b -= rho_inv * rho_inv * VectorTools::dyadic_product(delta_m_q, grad_w_q[0]);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_d =
      param_b;
    param_d += VectorTools::transpose(param_b);
    param_d -= 2. / 3. * VectorTools::trace(param_b) *
               VectorTools::identity<dim, dealii::VectorizedArray<number>>();
    param_d *= this->comp_flow_data.dynamic_viscosity;


    for (unsigned int i = 0; i < dim; ++i)
      {
        viscous_differential_change[i + 1] = param_c[i] + param_d[i];
      }

    /* change in energy density flux */
    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> tau_temp =
      this->calculator_functions.calculate_viscous_stress_tensor(grad_v_q_temp);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> tau;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        tau[i][j] = tau_temp[i][j];
    // density part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> rho_energy_density;
    rho_energy_density += VectorTools::matrix_vector_product(param_c, m_q) * rho_inv;
    rho_energy_density -=
      rho_inv * rho_inv * VectorTools::matrix_vector_product(tau, m_q) * delta_w_q[0];


    rho_energy_density +=
      lambda_div_c *
      (-rho_inv * rho_inv * grad_w_q[dim + 1] * delta_w_q[0] +
       rho_inv * rho_inv * rho_inv * w_q[dim + 1] * delta_w_q[0] * grad_w_q[0] +
       grad_w_q[0] * w_q[dim + 1] * rho_inv * rho_inv * rho_inv * delta_w_q[0] -
       w_q[dim + 1] * rho_inv * rho_inv * grad_delta_w_q[0] +
       VectorTools::matrix_vector_product(grad_v_q, m_q) * rho_inv * rho_inv * delta_w_q[0]);

    rho_energy_density -=
      lambda_div_c * rho_inv * rho_inv *
      (rho_inv * delta_w_q[0] *
         VectorTools::matrix_vector_product(VectorTools::dyadic_product(v_q, grad_w_q[0]), m_q) -
       VectorTools::matrix_vector_product(VectorTools::dyadic_product(v_q, grad_delta_w_q[0]),
                                          m_q));


    rho_energy_density += lambda_div_c * rho_inv * rho_inv * delta_w_q[0] *
                          VectorTools::matrix_vector_product(grad_v_q, m_q);


    // momentum part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum_energy_density;
    momentum_energy_density += rho_inv * VectorTools::matrix_vector_product(param_d, m_q);
    momentum_energy_density += rho_inv * VectorTools::matrix_vector_product(tau, delta_m_q);
    momentum_energy_density -=
      lambda_div_c * rho_inv * VectorTools::matrix_vector_product(grad_v_q, delta_m_q);
    momentum_energy_density -=
      lambda_div_c * rho_inv * rho_inv * VectorTools::matrix_vector_product(grad_delta_m_q, m_q);
    momentum_energy_density +=
      lambda_div_c * rho_inv * rho_inv * rho_inv *
      VectorTools::matrix_vector_product(VectorTools::dyadic_product(delta_m_q, grad_w_q[0]), m_q);


    // energy density part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> energy_energy_density;
    energy_energy_density += rho_inv * grad_delta_w_q[dim + 1];
    energy_energy_density -= grad_w_q[0] * rho_inv * rho_inv * delta_w_q[dim + 1];
    energy_energy_density *= lambda_div_c;

    // build the sum
    viscous_differential_change[dim + 1] =
      rho_energy_density + momentum_energy_density + energy_energy_density;
    return viscous_differential_change;
  }

  template <int dim, typename number>
  typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesGradType
  CompressibleFlowOperatorImplicitBase<dim, number>::
    calculate_jacobian_viscous_numerical_flux_jump_term(
      const std::pair<ConservedVariablesType, ConservedVariablesType> &delta_w_q,
      const Tensor<1, dim, VectorizedArray<number>>                   &normal,
      VectorizedArray<number>                                          penalty_parameter) const
  {
    return VectorTools::matrix_matrix_product(
      penalty_parameter * this->comp_flow_data.dynamic_viscosity /
        this->comp_flow_data.reference_density *
        VectorTools::identity<dim + 2, dealii::VectorizedArray<number>>(),
      VectorTools::dyadic_product(delta_w_q.first - delta_w_q.second, normal));
  }

  template <int dim, typename number>
  typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesGradType
  CompressibleFlowOperatorImplicitBase<dim, number>::
    calculate_jacobian_convective_numerical_flux_jump_term(
      const std::pair<ConservedVariablesType, ConservedVariablesType> &w_q,
      const std::pair<ConservedVariablesType, ConservedVariablesType> &delta_w_q,
      const Tensor<1, dim, VectorizedArray<number>>                   &normal) const
  {
    // define aliases
    const ConservedVariablesType &w_m       = w_q.first;
    const ConservedVariablesType &w_p       = w_q.second;
    const ConservedVariablesType &delta_w_m = delta_w_q.first;
    const ConservedVariablesType &delta_w_p = delta_w_q.second;
    // jump
    std::function<
      ConservedVariablesType(const ConservedVariablesType &,
                             const ConservedVariablesType &,
                             const ConservedVariablesType &,
                             const ConservedVariablesType &,
                             const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &)>
                                                                compute_lambda_times_jump;
    Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>> flux;

    if (this->comp_flow_data.numerical_flux_type == "lax_friedrichs_exact")
      {
        compute_lambda_times_jump =
          [this](const ConservedVariablesType                                  &w_p,
                 const ConservedVariablesType                                  &w_m,
                 const ConservedVariablesType                                  &delta_w_p,
                 const ConservedVariablesType                                  &delta_w_m,
                 const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal)
          -> ConservedVariablesType {
          const auto velocity_m = this->calculator_functions.calculate_velocity(w_m);
          const auto velocity_p = this->calculator_functions.calculate_velocity(w_p);

          const auto pressure_m =
            this->calculator_functions.calculate_pressure(w_m, this->comp_flow_data.gamma);
          const auto pressure_p =
            this->calculator_functions.calculate_pressure(w_p, this->comp_flow_data.gamma);

          const auto lambda =
            std::max(std::abs(velocity_p * normal) +
                       std::sqrt(this->comp_flow_data.gamma * pressure_p * (1. / w_p[0])),
                     std::abs(velocity_m * normal) +
                       std::sqrt(this->comp_flow_data.gamma * pressure_m * (1. / w_m[0])));
          return lambda * (delta_w_m - delta_w_p);
        };
      }
    else
      {
        compute_lambda_times_jump =
          [this](const ConservedVariablesType &w_p,
                 const ConservedVariablesType &w_m,
                 const ConservedVariablesType &delta_w_p,
                 const ConservedVariablesType &delta_w_m,
                 const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &)
          -> ConservedVariablesType {
          const auto velocity_m = this->calculator_functions.calculate_velocity(w_m);
          const auto velocity_p = this->calculator_functions.calculate_velocity(w_p);

          const auto pressure_m =
            this->calculator_functions.calculate_pressure(w_m, this->comp_flow_data.gamma);
          const auto pressure_p =
            this->calculator_functions.calculate_pressure(w_p, this->comp_flow_data.gamma);

          const auto lambda =
            0.5 *
            std::sqrt(std::max(velocity_p.norm_square() +
                                 std::abs(this->comp_flow_data.gamma * pressure_p * (1. / w_p[0])),
                               velocity_m.norm_square() + std::abs(this->comp_flow_data.gamma *
                                                                   pressure_m * (1. / w_m[0]))));
          return lambda * (delta_w_m - delta_w_p);
        };
      }

    const number epsilon     = 1e-3;
    const number epsilon_inv = 1. / 1e-3;
    if (this->comp_flow_data.linearization_jump_convective_flux == "complete_fd")
      {
        flux -= VectorTools::dyadic_product(0.5 * epsilon_inv *
                                              compute_lambda_times_jump(w_p, w_m, w_p, w_m, normal),
                                            normal);
        flux += VectorTools::dyadic_product(0.5 * epsilon_inv *
                                              compute_lambda_times_jump(w_p + epsilon * delta_w_p,
                                                                        w_m + epsilon * delta_w_m,
                                                                        w_p + epsilon * delta_w_p,
                                                                        w_m + epsilon * delta_w_m,
                                                                        normal),
                                            normal);
      }
    else if (this->comp_flow_data.linearization_jump_convective_flux == "lambda_fd")
      {
        flux += VectorTools::dyadic_product(
          0.5 * compute_lambda_times_jump(w_p, w_m, delta_w_p, delta_w_m, normal), normal);

        auto helper = compute_lambda_times_jump(
          w_p + epsilon * delta_w_p, w_m + epsilon * delta_w_m, w_p, w_m, normal);
        helper -= compute_lambda_times_jump(w_p, w_m, w_p, w_m, normal);
        flux += VectorTools::dyadic_product(0.5 / epsilon * helper, normal);
      }
    else if (this->comp_flow_data.linearization_jump_convective_flux == "analytic" &&
             this->comp_flow_data.numerical_flux_type == "lax_friedrichs_exact")
      {
        flux += VectorTools::dyadic_product(
          0.5 * compute_lambda_times_jump(w_p, w_m, delta_w_p, delta_w_m, normal), normal);

        const auto linearize_speed_of_sound =
          [this](const ConservedVariablesType          &w_q,
                 const ConservedVariablesType          &delta_w_q,
                 const dealii::VectorizedArray<number> &c,
                 const dealii::VectorizedArray<number> &) -> dealii::VectorizedArray<number> {
          dealii::Tensor<1, dim, dealii::VectorizedArray<number>> m_q;
          dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_m_q;
          for (unsigned int i = 0; i < dim; ++i)
            {
              m_q[i]       = w_q[i + 1];
              delta_m_q[i] = delta_w_q[i + 1];
            }
          dealii::VectorizedArray<number> rho_inv = 1. / w_q[0];
          dealii::VectorizedArray<number> lin_c   = delta_w_q[0] / (2.0 * c) *
                                                  this->comp_flow_data.gamma * rho_inv * rho_inv *
                                                  rs_div_c * (rho_inv * m_q * m_q - w_q[dim + 1]);
          lin_c -= 1. / (2.0 * c) * this->comp_flow_data.gamma * rho_inv * rho_inv * rs_div_c *
                   m_q * delta_m_q;
          lin_c +=
            1. / (2.0 * c) * this->comp_flow_data.gamma * rho_inv * rs_div_c * delta_w_q[dim + 1];
          return lin_c;
        };

        const auto norm_lin_velocity =
          [this](const ConservedVariablesType                                  &w_q,
                 const ConservedVariablesType                                  &delta_w_q,
                 const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal)
          -> dealii::VectorizedArray<number> {
          dealii::VectorizedArray<number>                         rho_inv = 1. / w_q[dim + 1];
          dealii::Tensor<1, dim, dealii::VectorizedArray<number>> u =
            this->calculator_functions.calculate_velocity(w_q);
          dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_m_q;
          for (unsigned int i = 0; i < dim; ++i)
            delta_m_q[i] = delta_w_q[i + 1];
          dealii::Tensor<1, dim, dealii::VectorizedArray<number>> lin_velocity =
            delta_m_q * rho_inv - u * rho_inv * delta_w_q[0];
          return lin_velocity * normal;
        };

        const auto signum =
          [](const dealii::VectorizedArray<number> &a) -> const dealii::VectorizedArray<number> {
          return dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
            a, dealii::VectorizedArray<number>(0.0), 1.0, -1.0);
        };

        const auto velocity_m = this->calculator_functions.calculate_velocity(w_m);
        const auto velocity_p = this->calculator_functions.calculate_velocity(w_p);

        const auto pressure_m =
          this->calculator_functions.calculate_pressure(w_m, this->comp_flow_data.gamma);
        const auto pressure_p =
          this->calculator_functions.calculate_pressure(w_p, this->comp_flow_data.gamma);

        const auto speed_of_sound_m = std::sqrt(this->comp_flow_data.gamma * pressure_m / w_m[0]);
        const auto speed_of_sound_p = std::sqrt(this->comp_flow_data.gamma * pressure_p / w_p[0]);

        const auto lin_lambda_p =
          signum(velocity_p * normal) * norm_lin_velocity(w_p, delta_w_p, normal) +
          linearize_speed_of_sound(w_p, delta_w_p, speed_of_sound_p, pressure_p);
        const auto lin_lambda_m =
          signum(velocity_m * normal) * norm_lin_velocity(w_m, delta_w_m, normal) +
          linearize_speed_of_sound(w_m, delta_w_m, speed_of_sound_m, pressure_m);

        const auto lin_lambda =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(velocity_p.norm() +
                                                                                 speed_of_sound_p,
                                                                               velocity_m.norm() +
                                                                                 speed_of_sound_m,
                                                                               lin_lambda_p,
                                                                               lin_lambda_m);

        for (unsigned int i = 0; i < dim + 2; ++i)
          flux[i] += 0.5 * lin_lambda * (w_m[i] - w_p[i]) * normal;
      }
    else
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "The provided linearization scheme of the jump operator term in the convective"
            " numerical flux is not supported."));
      }
    return flux;
  }

  template <int dim, typename number>
  std::tuple<typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesType,
             typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesGradType,
             typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesType,
             typename CompressibleFlowOperatorImplicitBase<dim, number>::ConservedVariablesGradType>
  CompressibleFlowOperatorImplicitBase<dim, number>::get_adjacent_jacobian_face_values_at_boundary(
    const Point<dim, VectorizedArray<number>>     &q_point,
    const Tensor<1, dim, VectorizedArray<number>> &normal,
    unsigned int                                   boundary_id,
    const ConservedVariablesType                  &w_m,
    const ConservedVariablesType                  &delta_w_m,
    const ConservedVariablesGradType              &grad_w_m,
    const ConservedVariablesGradType              &grad_delta_w_m) const
  {
    ConservedVariablesType     w_p, delta_w_p;
    ConservedVariablesGradType grad_w_p, grad_delta_w_p;
    if (this->slip_wall_boundaries.contains(boundary_id))
      {
        // homogeneous Neumann
        auto rho_u_dot_n = w_m[1] * normal[0];
        for (unsigned int d = 1; d < dim; ++d)
          rho_u_dot_n += w_m[1 + d] * normal[d];
        w_p[0]            = w_m[0];
        grad_w_p[0]       = -(grad_w_m[0]);
        grad_delta_w_p[0] = -grad_delta_w_m[0];
        // symmetry
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
            grad_w_p[d + 1] =
              grad_w_m[d + 1] - 2. * scalar_product(grad_w_m[d + 1], normal) * normal;
            grad_delta_w_p[d + 1] =
              grad_delta_w_m[d + 1] - 2. * scalar_product(grad_delta_w_m[d + 1], normal) * normal;
          }
        // homogeneous Neumann
        grad_w_p[dim + 1]       = -(grad_w_m[dim + 1]);
        grad_delta_w_p[dim + 1] = -grad_delta_w_m[dim + 1];
        w_p[dim + 1]            = w_m[dim + 1];

        delta_w_p[0] = delta_w_m[0];
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_momentum;
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            delta_momentum[i - 1] = delta_w_m[i];
          }
        const auto matrix =
          -2.0 *
          VectorTools::dyadic_product<dim, dim, dealii::VectorizedArray<number>>(normal, normal);
        const auto helper =
          VectorTools::matrix_vector_product<dim, dim, dealii::VectorizedArray<number>>(
            matrix, delta_momentum);
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            delta_w_p[i] = helper[i - 1] + delta_w_m[i];
          }
        delta_w_p[dim + 1] = delta_w_m[dim + 1];
      }
    else if (this->no_slip_adiabatic_wall_boundaries.contains(boundary_id))
      {
        // homogeneous Neumann
        w_p[0]            = w_m[0];
        grad_w_p[0]       = -(grad_w_m[0]);
        delta_w_p[0]      = delta_w_m[0];
        grad_delta_w_p[0] = -grad_delta_w_m[0];
        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1]            = 0.;
            grad_w_p[d + 1]       = grad_w_m[d + 1];
            delta_w_p[d + 1]      = 0.0;
            grad_delta_w_p[d + 1] = grad_delta_w_m[d + 1];
          }
        // homogeneous Neumann
        grad_w_p[dim + 1]       = -(grad_w_m[dim + 1]);
        w_p[dim + 1]            = w_m[dim + 1];
        delta_w_p[dim + 1]      = delta_w_m[dim + 1];
        grad_delta_w_p[dim + 1] = -grad_delta_w_m[dim + 1];
      }
    else if (this->inflow_boundaries.contains(boundary_id))
      {
        // Dirichlet
        w_p = VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
          *this->inflow_boundaries.find(boundary_id)->second, q_point);

        grad_w_p = grad_w_m;
        // delta_w_p is zero at Dirichlet boundaries. Hence, nothing needs to be done here.

        grad_delta_w_p = grad_delta_w_m;
      }
    else if (this->subsonic_outflow_fixed_pressure.contains(boundary_id))
      {
        // homogeneous Neumann
        w_p            = w_m;
        grad_w_p       = -grad_w_m;
        delta_w_p      = delta_w_m;
        grad_delta_w_p = -grad_delta_w_m;

        // Dirichlet
        auto p_dyn = VectorizedArray<double>(0.);
        for (unsigned int i = 1; i < dim + 1; ++i)
          p_dyn += w_m[i] * w_m[i];

        p_dyn /= (w_m[0] * 2.);
        w_p[dim + 1] =
          VectorTools::evaluate_function_at_vectorized_points<dim, number>(
            *this->subsonic_outflow_fixed_pressure.find(boundary_id)->second, q_point, dim + 1) /
            (this->comp_flow_data.gamma - 1) +
          p_dyn;
        grad_w_p[dim + 1]       = grad_w_m[dim + 1];
        grad_delta_w_p[dim + 1] = grad_delta_w_m[dim + 1];

        dealii::VectorizedArray<number>                         rho_inv = 1.0 / w_m[0];
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum;
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_momentum;
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            momentum[i - 1]       = w_m[i];
            delta_momentum[i - 1] = delta_w_m[i];
          }
        delta_w_p[dim + 1] = momentum.norm() * rho_inv * rho_inv * rho_inv * delta_w_m[0] +
                             rho_inv * momentum * delta_momentum;
      }
    else if (this->subsonic_outflow_fixed_energy.contains(boundary_id))
      {
        // homogeneous Neumann
        w_p            = w_m;
        grad_w_p       = -grad_w_m;
        delta_w_p      = delta_w_m;
        grad_delta_w_p = -grad_delta_w_m;

        // Dirichlet
        w_p[dim + 1] = VectorTools::evaluate_function_at_vectorized_points<dim, number>(
          *this->subsonic_outflow_fixed_energy.find(boundary_id)->second, q_point, dim + 1);

        grad_w_p[dim + 1]       = grad_w_m[dim + 1];
        delta_w_p[dim + 1]      = 0.0;
        grad_delta_w_p[dim + 1] = grad_delta_w_m[dim + 1];
      }
    else
      AssertThrow(false,
                  ExcMessage("Unknown boundary id, did "
                             "you set a boundary condition for "
                             "this part of the domain boundary?"));
    return {w_p, grad_w_p, delta_w_p, grad_delta_w_p};
  }



} // namespace MeltPoolDG::Flow
