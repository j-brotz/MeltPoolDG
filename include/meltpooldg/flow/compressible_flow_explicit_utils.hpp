/**
 * @brief A collection of helper functions that might be useful when solving the compressible
 * Navier-Stokes equations with an explicit time stepping strategy.
 */

#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

namespace MeltPoolDG::Flow
{
  template <typename evaluator_type,
            int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType>
  concept CellEvaluatorType =
    std::is_base_of_v<dealii::FECellIntegrator<dim, n_components, number, VectorizedArrayType>,
                      evaluator_type> ||
    std::is_base_of_v<dealii::FEPointEvaluation<n_components, dim, dim, VectorizedArrayType>,
                      evaluator_type>;

  template <typename evaluator_type,
            int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType>
  concept FaceEvaluatorType =
    std::is_base_of_v<dealii::FEFaceIntegrator<dim, n_components, number, VectorizedArrayType>,
                      evaluator_type> ||
    std::is_base_of_v<dealii::FEFacePointEvaluation<n_components, dim, dim, VectorizedArrayType>,
                      evaluator_type>;

  /**
   * Kernel of the local cell applier for the right-hand side function. This function computes the
   * cell integral contribution to the right hand side for the quadrature point index and the
   * corresponding FE evaluator.
   *
   * @param evaluator FE-evaluator object reinitialized on the current cell batch.
   * @param q Index of the quadrature point.
   * @param constant_body_force Value of the body force. If the body force is not constant the
   * pointer must be set to nullptr.
   * @param convective_terms Collection of convective term computations for the compressible Navier-Stokes equations.
   * @param viscous_terms Collection of viscous term computations for the compressible Navier-Stokes equations.
   * @param flow_scratch_data Struct providing the relevant data required by all compressible flow solvers.
   *
   * @return Tuple, containing the flux, weighted with the value of the test function, as first
   * argument, and the flux, weighted with the gradient of the test function, as second argument.
   */
  template <int dim,
            typename number,
            CellEvaluatorType<dim,
                              dim + 2,
                              number,
                              dealii::VectorizedArray<number>> Integrator,
            bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<CompressibleFlowTypes::ConservedVariablesType<dim, number>,
               CompressibleFlowTypes::ConservedVariablesGradType<dim, number>>
    rhs_cell_integral_kernel(
      const Integrator                                              &evaluator,
      const unsigned int                                             q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> *constant_body_force,
      const CompressibleFlowConvectiveKernels<dim, number>          &convective_terms,
      const CompressibleFlowViscousKernels<dim, number>             &viscous_terms,
      const CompressibleFlowScratchData<dim, number>                &flow_scratch_data)
  {
    const auto w_q = evaluator.get_value(q);

    auto flux = convective_terms.template calculate_convective_flux<is_gas_phase>(w_q);

    // TODO: introduce a template parameter 'is_viscous'
    if (flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0)
      {
        const auto grad_w_q = evaluator.get_gradient(q);
        flux -= viscous_terms.template calculate_viscous_flux<is_gas_phase>(w_q, grad_w_q);
      }

    dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>> forcing;

    if (flow_scratch_data.body_force.get() != nullptr)
      {
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> force =
          constant_body_force ?
            *constant_body_force :
            VectorTools::evaluate_function_at_vectorized_points(*flow_scratch_data.body_force,
                                                                evaluator.quadrature_point(q));
        for (unsigned int d = 0; d < dim; ++d)
          forcing[d + 1] = w_q[0] * force[d];
        for (unsigned int d = 0; d < dim; ++d)
          forcing[dim + 1] += force[d] * w_q[d + 1];
      }

    return {forcing, flux};
  }

  /**
   * Kernel of the local inner face applier for the right-hand side function. This function
   * computes the face integral contribution of inner faces to the right hand side for the
   * quadrature point index and the corresponding FE evaluator.
   *
   * @param evaluator_m FE-evaluator object reinitialized on the current (inside) face batch.
   * @param evaluator_p FE-evaluator object reinitialized on the current (outside) face batch.
   * @param q Index of the quadrature point.
   * @param penalty_parameter Value of the symmetric interior penalty parameter on the face.
   * @param convective_terms Collection of convective term computations for the compressible Navier-Stokes equations.
   * @param viscous_terms Collection of viscous term computations for the compressible Navier-Stokes equations.
   * @param dynamic_viscosity Dynamic viscosity.
   *
   * @return Tuple, which containing the fluxes for the inside and outside faces, weighted with
   * the value of the test functions, as first two arguments, and the fluxes for the inside and
   * outside faces, weighted with the gradient of the test functions, as the third and fourth
   * argument.
   */
  template <int dim,
            typename number,
            FaceEvaluatorType<dim,
                              dim + 2,
                              number,
                              dealii::VectorizedArray<number>> Integrator,
            bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<CompressibleFlowTypes::ConservedVariablesType<dim, number>,
               CompressibleFlowTypes::ConservedVariablesType<dim, number>,
               CompressibleFlowTypes::ConservedVariablesGradType<dim, number>,
               CompressibleFlowTypes::ConservedVariablesGradType<dim, number>>
    rhs_face_integral_kernel(const Integrator               &evaluator_m,
                             const Integrator               &evaluator_p,
                             const unsigned int              q,
                             dealii::VectorizedArray<number> penalty_parameter,
                             const CompressibleFlowConvectiveKernels<dim, number> &convective_terms,
                             const CompressibleFlowViscousKernels<dim, number>    &viscous_terms,
                             const number dynamic_viscosity)
  {
    auto numerical_flux =
      convective_terms.template calculate_convective_numerical_flux<is_gas_phase>(evaluator_m.get_value(q),
                                                           evaluator_p.get_value(q),
                                                           evaluator_m.normal_vector(q));

    if (dynamic_viscosity > 0)
      numerical_flux -= viscous_terms.template calculate_viscous_numerical_flux<is_gas_phase>(evaluator_m.get_value(q),
                                                                       evaluator_p.get_value(q),
                                                                       evaluator_m.get_gradient(q),
                                                                       evaluator_p.get_gradient(q),
                                                                       evaluator_m.normal_vector(q),
                                                                       penalty_parameter);

    std::pair<CompressibleFlowTypes::ConservedVariablesGradType<dim, number>,
              CompressibleFlowTypes::ConservedVariablesGradType<dim, number>>
      viscous_numerical_flux;

    // interior penalty
    if (dynamic_viscosity > 0)
      {
        viscous_numerical_flux =
          viscous_terms.template calculate_viscous_numerical_flux_gradient<is_gas_phase>(evaluator_m.get_value(q),
                                                                  evaluator_p.get_value(q),
                                                                  evaluator_m.normal_vector(q));
      }

    return {-numerical_flux,
            numerical_flux,
            viscous_numerical_flux.first,
            viscous_numerical_flux.second};
  }

  /**
   * Kernel of the local boundary face applier for the right-hand side function. This function
   * computes the face integral contribution of boundary faces to the right hand side for the
   * quadrature point index and the corresponding FE evaluator.
   *
   * @param evaluator_m FE-evaluator object reinitialized on the current (inner) face batch.
   * @param q Index of the quadrature point.
   * @param boundary_id Boundary ID of the considered boundary face.
   * @param penalty_parameter Value of the symmetric interior penalty parameter on the face.
   * @param convective_terms Collection of convective term computations for the compressible Navier-Stokes equations.
   * @param viscous_terms Collection of viscous term computations for the compressible Navier-Stokes equations.
   * @param flow_scratch_data Struct providing the relevant data required by all compressible flow solvers.
   *
   * @return Tuple, containing the flux for the boundary face, weighted with the value of the test
   * function, as first argument, and the flux for the boundary face, weighted with the gradient
   * of the test function, as second argument.
   */
  template <int dim,
            typename number,
            FaceEvaluatorType<dim,
                              dim + 2,
                              number,
                              dealii::VectorizedArray<number>> Integrator,
            bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<CompressibleFlowTypes::ConservedVariablesType<dim, number>,
               CompressibleFlowTypes::ConservedVariablesGradType<dim, number>>
    rhs_boundary_face_integral_kernel(
      const Integrator                                     &evaluator_m,
      const unsigned int                                    q,
      const dealii::types::boundary_id                      boundary_id,
      const dealii::VectorizedArray<number>                 penalty_parameter,
      const CompressibleFlowConvectiveKernels<dim, number> &convective_terms,
      const CompressibleFlowViscousKernels<dim, number>    &viscous_terms,
      const CompressibleFlowScratchData<dim, number>       &flow_scratch_data)
  {
    const auto w_m      = evaluator_m.get_value(q);
    const auto normal   = evaluator_m.normal_vector(q);
    const auto grad_w_m = evaluator_m.get_gradient(q);

    auto [w_p, grad_w_p] =
      flow_scratch_data.boundary_conditions.get_boundary_face_value_and_gradient(
        evaluator_m.quadrature_point(q),
        normal,
        boundary_id,
        w_m,
        grad_w_m,
        flow_scratch_data.flow_data,
        is_gas_phase);

    auto flux = convective_terms.template calculate_convective_numerical_flux<is_gas_phase>(w_m, w_p, normal);

    if (flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0)
      flux -= viscous_terms.template calculate_viscous_numerical_flux<is_gas_phase>(
        w_m, w_p, grad_w_m, grad_w_p, normal, penalty_parameter);

    CompressibleFlowTypes::ConservedVariablesGradType<dim, number> numerical_flux_gradient;

    if (flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0)
      {
        numerical_flux_gradient =
          viscous_terms.template calculate_viscous_numerical_flux_gradient<is_gas_phase>(w_m, w_p, normal).first;
      }

    return {-flux, numerical_flux_gradient};
  }
} // namespace MeltPoolDG::Flow
