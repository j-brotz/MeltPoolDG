#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_base.hpp>
#include <meltpooldg/flow/dg_generic_convection_diffusion_worker.hpp>
#include <meltpooldg/flow/generic_staggered_loop_operator.hpp>
#include <meltpooldg/time_integration/explicit_low_storage_runge_kutta_integrator.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <functional>
#include <memory>
#include <tuple>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Operator for the matrix-free evaluation of a compressible single-phase flow cutDG
   * formulation for explicit time integration.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   * @tparam is_viscous Indicates whether the flow is viscous.
   */
  template <int dim, typename number, bool is_viscous = true>
  class DGCompressibleFlowOperatorExplicit final
    : public DGCompressibleFlowOperatorBase<dim, number>
  {
  public:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    using ConvectionDiffusionOperator =
      DGConvectionDiffusionOperator<dim,
                                    number,
                                    CompressibleConvectiveFlux<dim, number>,
                                    CompressibleDiffusiveFlux<dim, number>>;

    using ConvectionOperator =
      DGConvectionOperator<dim, number, CompressibleConvectiveFlux<dim, number>>;

    using TriaIterator = dealii::TriaIterator<dealii::CellAccessor<dim>>;

    /**
     * @brief Constructor.
     *
     * @param flow_scratch_data Reference to the flow scratch data object (usually owned by the
     * corresponding operation class).
     * @param external_forces Pointer to a struct implementing external forces acting on the fluid.
     */
    explicit DGCompressibleFlowOperatorExplicit(
      CompressibleFlowScratchData<dim, number> &flow_scratch_data);

    /**
     * @brief Advances solver by a single time step.
     *
     * This function performs a single explicit time step of size @p time_step starting from the
     * solution at time @p time.
     *
     * @note The function does not take care about updating the solution history object or similar
     * operations which are not directly related to the integration. It **only** advances the
     * solution by a single time step starting from the current solution in the solution history
     * object of the @ref flow_scratch_data object.
     */
    void
    advance_time_step(number time, number time_step) override;

    /**
     * @brief Reinitialize the internal data structures.
     *
     * The reinitialization includes setting a new required size for the solution history object
     * according to the demands of the used time integrator.
     */
    void
    reinit() override;

    /**
     * @brief Computes the value of the function f(y) for the compressible Navier-Stokes equations of the
     * form y' = f(y).
     *
     * From a discretization perspective, f(y) is given by f(y) = M^(-1) * F(y),
     * where M is the mass matrix and F(y) is the sum of all flux contributions: F_v + F_c + F_rhs.
     *
     * @param time The current time at which the function is evaluated.
     * @param dst Vector where the computed value of f(y) is stored.
     * @param src The solution vector, y, at the current time.
     * @param func A function to be executed after f(y) has been computed. This function is applied
     * to the resulting vector in @p dst.
     */
    void
    apply_operator(number                                                 time,
                   number                                                 time_step,
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func) const;

    void
    add_external_force(std::shared_ptr<ExternalFlowForce<dim, number>> external_force,
                       std::shared_ptr<ExternalFlowForceJacobian<dim, number>>) override;

  private:
    /// Scratch data for compressible flows
    CompressibleFlowScratchData<dim, number> &flow_scratch_data;

    /// Time integrator class used for the time integration.
    TimeIntegration::LowStorageExplicitRungeKuttaIntegrator<number> time_integrator;

    /// This pointer may hold an instance of an external fluid force contribution
    /// (e.g., gravity, body forces, or user - defined source terms)
    std::vector<std::shared_ptr<ExternalFlowForce<dim, number>>> external_forces;

    /// Current time step size
    mutable number current_time_step;

    ///
    GenericStaggeredLoopOperator<dim, CompressibleFlow::n_conserved_variables<dim>, number>
      generic_operator;

    std::tuple<CompressibleFlow::SourceType<dim, number>, CompressibleFlow::FluxType<dim, number>>
    cell_quad_operation(
      const unsigned int                                                   cell_batch_index,
      const CompressibleFlow::ConservedVariablesType<dim, number>         &w_q,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number> &grad_w_q,
      const dealii::Point<dim, VectorizedArrayType>                       &quadrature_point) const;

    std::tuple<CompressibleFlow::FaceFluxType<dim, number>,
               CompressibleFlow::FaceGradientFluxType<dim, number>,
               CompressibleFlow::FaceFluxType<dim, number>,
               CompressibleFlow::FaceGradientFluxType<dim, number>>
    face_quad_operation(
      const std::array<unsigned int, VectorizedArrayType::size()>         &cell_ids_m,
      const std::array<unsigned int, VectorizedArrayType::size()>         &cell_ids_p,
      const CompressibleFlow::ConservedVariablesType<dim, number>         &w_m,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number> &grad_w_m,
      const CompressibleFlow::ConservedVariablesType<dim, number>         &w_p,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number> &grad_w_p,
      const dealii::Tensor<1, dim, VectorizedArrayType>                   &normal,
      const dealii::Point<dim, VectorizedArrayType>                       &quadrature_point) const;

    std::tuple<CompressibleFlow::FaceFluxType<dim, number>,
               CompressibleFlow::FaceGradientFluxType<dim, number>>
    boundary_face_quad_operation(
      const std::array<unsigned int, VectorizedArrayType::size()>         &cell_ids_m,
      const CompressibleFlow::ConservedVariablesType<dim, number>         &w_m,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number> &grad_w_m,
      const dealii::Tensor<1, dim, VectorizedArrayType>                   &normal,
      const dealii::Point<dim, VectorizedArrayType>                       &quadrature_point,
      const dealii::types::boundary_id                                     boundary_id) const;
  };
} // namespace MeltPoolDG::Flow
