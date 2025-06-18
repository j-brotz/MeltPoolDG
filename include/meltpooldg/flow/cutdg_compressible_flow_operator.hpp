#pragma once

#include <deal.II/fe/fe_system.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>

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

    /// Object for the convective term evaluations
    const CompressibleFlowConvectiveKernels<dim, number> convective_terms;

    /// Object for the viscous term evaluations
    const CompressibleFlowViscousKernels<dim, number> viscous_terms;

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

    /// Inverse time step size
    mutable number inv_time_step = 0.;

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
