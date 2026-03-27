#pragma once

#include <deal.II/fe/fe_system.h>

#include <meltpooldg/compressible_flow/dg_generic_convection_diffusion_worker.hpp>
#include <meltpooldg/compressible_flow/explicit_utils.hpp>
#include <meltpooldg/compressible_flow/flow_scratch_data.hpp>
#include <meltpooldg/compressible_flow/kernels.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Operator for the matrix-free evaluation of a compressible single-phase flow cutDG
   * formulation.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   * @tparam is_viscous Indicates whether the flow is viscous.
   */
  template <int dim, typename number, bool is_viscous = true>
  class CutDGCompressibleFlowOperator
  {
  public:
    using VectorType            = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

    using ConservedVariablesType = dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>;
    using ConservedVariablesGradType =
      dealii::Tensor<1, dim + 2, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;

    using ConvectionDiffusionOperator =
      DGConvectionDiffusionOperator<dim,
                                    number,
                                    CompressibleConvectiveFlux<dim, number>,
                                    CompressibleDiffusiveFlux<dim, number>>;

    using ConvectionOperator =
      DGConvectionOperator<dim, number, CompressibleConvectiveFlux<dim, number>>;

    /**
     * @brief Constructor.
     *
     * Initializes the operators, integrators, and mapping objects needed to compute
     * the matrix-free evaluation of a compressible single-phase flow cutDG formulation.
     *
     * @param flow_scratch_data Flow scratch data object holding all relevant compressible flow data
     * required by the operator.
     * @param mapping_info_surface_in dealii::NonMatching::MappingInfo object, provides the mapping
     * information computation and mapping data storage of the surface.
     * @param mapping_info_cells_in Vector of dealii::NonMatching::MappingInfo objects, provides
     * the mapping information computation and mapping data storage of the cells on the
     * inner subdomain and the outer subdomain, respectively.
     * @param mapping_info_faces_in Vector of dealii::NonMatching::MappingInfo objects, provides
     * the mapping information computation and mapping data storage of the faces on the
     * inner subdomain and the outer subdomain, respectively.
     */
    explicit CutDGCompressibleFlowOperator(
      CompressibleFlowScratchData<dim, number> &flow_scratch_data,
      const MappingInfoType                    &mapping_info_surface_in,
      const MappingInfoVectorType              &mapping_info_cells_in,
      const MappingInfoVectorType              &mapping_info_faces_in);

    /**
     * @brief Set the inflow field function in the case of an unfitted inflow boundary.
     *
     * @param inflow_function Function which describes the inflow field at the unfitted inflow boundary.
     */
    void
    set_inflow_field_unfitted_boundary(std::shared_ptr<dealii::Function<dim>> &inflow_function);

    /**
     * @brief Set the velocity function in the case of an unfitted (rigid) moving object.
     *
     * @param velocity_function Scalar function which describes the velocity of an unfitted (rigid) moving object
     */
    void
    set_unfitted_object_velocity(std::shared_ptr<dealii::Function<dim>> &velocity_function);

    /**
     * @brief Evaluate and submit the right-hand side cell integral contributions at a given quadrature point.
     *
     * This function can be used for both cut and non-cut elements, depending on the templated @p Integrator type.
     *
     * @tparam Integrator Type of the matrix-free cell integrator providing the interface for evaluating
     * and submitting values at quadrature points.
     *
     * @param phi  Cell integrator used to access solution values and to submit
     * the local integral contributions at quadrature point @p q.
     * @param flow_scratch_data Flow scratch data object holding all relevant compressible flow data.
     * @param constant_body_force Value of the body force. If the body force is not constant the
     * pointer must be set to nullptr.
     * @param q Index of the quadrature point.
     */
    template <CellEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator>
    void
    do_cell_integral_rhs(
      Integrator                                                    &phi,
      const CompressibleFlowScratchData<dim, number>                &flow_scratch_data,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> *constant_body_force,
      const unsigned int                                             q) const;

    /**
     * @brief Evaluate and submit the right-hand side cut surface integral contributions at a given quadrature point.
     *
     * @param phi  Surface integrator used to access solution values and to submit
     * the local integral contributions at quadrature point @p q.
     * @param interior_penalty_parameter Interior penalty parameter for symmetric interior penalty flux.
     * @param flow_scratch_data Flow scratch data object holding all relevant compressible flow data.
     * @param q Index of the quadrature point.
     */
    void
    do_surface_integral_rhs(
      dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>> &phi,
      const dealii::VectorizedArray<number>          &interior_penalty_parameter,
      const CompressibleFlowScratchData<dim, number> &flow_scratch_data,
      const unsigned int                              q) const;

    /**
     * @brief Evaluate and submit the right-and side face integral contributions at a given quadrature point.
     *
     * This function can be used for both cut and non-cut faces, depending on the templated @p Integrator type.
     *s
     * @tparam Integrator Type of the matrix-free face integrator providing the interface for evaluating
     * and submitting values at quadrature points.
     *
     * @param phi_m Face integrator used to access solution values and to submit
     * the local integral contributions at quadrature point @p q at the interior face.
     * @param phi_p Face integrator used to access solution values and to submit
     * the local integral contributions at quadrature point @p q at the exterior face.
     * @param interior_penalty_parameter Interior penalty parameter for symmetric interior penalty flux.
     * @param flow_scratch_data Flow scratch data object holding all relevant compressible flow data.
     * @param q Index of the quadrature point.
     */
    template <FaceEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator>
    void
    do_face_integral_rhs(Integrator                                     &phi_m,
                         Integrator                                     &phi_p,
                         const dealii::VectorizedArray<number>          &interior_penalty_parameter,
                         const CompressibleFlowScratchData<dim, number> &flow_scratch_data,
                         const unsigned int                              q) const;

    /**
     * @brief Evaluate and submit the right-hand side boundary face integral contributions at a given quadrature point.
     *
     * This function can be used for both cut and non-cut faces, depending on the templated @p Integrator type.
     *
     * @tparam Integrator Type of the matrix-free face integrator providing the interface for evaluating
     * and submitting values at quadrature points.
     *
     * @param phi Face integrator used to access solution values and to submit
     * the local integral contributions at quadrature point @p q at the boundary face.
     * @param interior_penalty_parameter Interior penalty parameter for symmetric interior penalty flux.
     * @param flow_scratch_data Flow scratch data object holding all relevant compressible flow data.
     * @param boundary_id Boundary id of the current face.
     * @param q Index of the quadrature point.
     */
    template <FaceEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator>
    void
    do_boundary_face_integral_rhs(Integrator                            &phi,
                                  const dealii::VectorizedArray<number> &interior_penalty_parameter,
                                  const CompressibleFlowScratchData<dim, number> &flow_scratch_data,
                                  const auto                                      boundary_id,
                                  const unsigned int                              q) const;

    /**
     * @brief Local applier for the cell integrals in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param cell_range Considered cell range.
     */
    void
    local_apply_cell_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Local applier for the face integrals in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_face_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the boundary face integral in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_boundary_face_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType                                  &dst,
                                  const VectorType                            &src,
                                  const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the cell integrals in the left-hand side matrix-vector product evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param cell_range Considered cell range.
     */
    void
    local_apply_cell_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Local applier for the face integrals in the left-hand side matrix-vector product evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_face_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the boundary face integrals in the left-hand side matrix-vector product
     * evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_boundary_face_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType                                  &dst,
                                  const VectorType                            &src,
                                  const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Function for the matrix-free right-hand side vector evaluation.
     *
     * @param time Current simulation time.
     * @param time_step Current time step size.
     * @param dst Vector where the computed right-hand side rhs(src) is stored.
     * @param src The solution vector at the current time.
     */
    void
    create_rhs(const number     &time,
               const number     &time_step,
               VectorType       &dst,
               const VectorType &src) const;

    /**
     * @brief Function for the matrix-free matrix-vector product evaluation.
     *
     * @param dst Vector where the computed evaluated vector-matrix product is stored.
     * @param src The solution vector at the current time.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    /// Scratch data for compressible flows
    CompressibleFlowScratchData<dim, number> &flow_scratch_data;

    /// Mapping information for integration over immersed boundaries
    const MappingInfoType &mapping_info_surface;

    /// Mapping information for integration over cut cells
    const MappingInfoVectorType &mapping_info_cells;

    /// Mapping information for integration over cut faces
    const MappingInfoVectorType &mapping_info_faces;

    /// FESystem object, required by FEPointEvaluation
    dealii::FESystem<dim> fe_point_temp;

    /// Number of DoFs per cell
    const unsigned int n_dofs_per_cell;

    /// Current time step size
    mutable number current_time_step = 0.;

    /// Current inverse time step size
    mutable number current_inv_time_step = 0.;

    /// Function, which describes the velocity of the unfitted object
    std::shared_ptr<dealii::Function<dim>> unfitted_object_velocity;

    /// Inflow function for unfitted inflow boundary
    std::shared_ptr<dealii::Function<dim>> unfitted_inflow;

    /**
     * @brief This function sets the corresponding values on the fictional outer face if the face is
     * located at an unfitted boundary.
     *
     * @param q_point Location of the quadrature points at which the values shall be computed.
     * @param w_m Conserved variables on the inner face.
     * @param w_p Location where the corresponding boundary values are stored.
     * @param grad_w_m Gradient of the conserved variables on the inner face.
     * @param grad_w_p Location where the corresponding gradients of the boundary values shall
     * be stored.
     */
    void
    get_adjacent_face_values_at_unfitted_boundary(
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      const ConservedVariablesType                              &w_m,
      ConservedVariablesType                                    &w_p,
      const ConservedVariablesGradType                          &grad_w_m,
      ConservedVariablesGradType                                &grad_w_p) const;
  };
} // namespace MeltPoolDG::Flow
