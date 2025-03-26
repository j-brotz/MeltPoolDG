#pragma once

#include <deal.II/fe/fe_system.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>

namespace MeltPoolDG::Flow
{
  template <unsigned int dim, typename number = double, bool is_viscous = true>
  class CutDGCompressibleFlowOperator
  {
    using VectorType            = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

  public:
    using ConservedVariablesType     = Tensor<1, dim + 2, VectorizedArray<number>>;
    using ConservedVariablesGradType = Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>>;


    /**
     * Constructor.
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
    CutDGCompressibleFlowOperator(CompressibleFlowScratchData<dim, number> &flow_scratch_data,
                                  const MappingInfoType                    &mapping_info_surface_in,
                                  const MappingInfoVectorType              &mapping_info_cells_in,
                                  const MappingInfoVectorType              &mapping_info_faces_in);

    /**
     * Set the inflow field function in the case of an unfitted inflow boundary.
     *
     * @param inflow_function Function which describes the inflow field at the unfitted inflow boundary.
     */
    void
    set_inflow_field_unfitted_boundary(std::shared_ptr<Function<dim>> &inflow_function);

    /**
     * Set the velocity function in the case of an unfitted (rigid) moving object.
     *
     * @param velocity_function Scalar function which describes the velocity of an unfitted (rigid) moving object
     */
    void
    set_unfitted_object_velocity(std::shared_ptr<Function<dim>> &velocity_function);

    /**
     * Local appliers for right-hand side evaluation.
     */
    void
    local_apply_cell_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_apply_face_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const;

    void
    local_apply_boundary_face_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType                                  &dst,
                                  const VectorType                            &src,
                                  const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * Local appliers for left-hand side matrix-vector product evaluation.
     */
    void
    local_apply_cell_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_apply_face_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const;

    void
    local_apply_boundary_face_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType                                  &dst,
                                  const VectorType                            &src,
                                  const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * Function for the matrix-free right-hand side vector evaluation.
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
     * Function for the matrix-free matrix-vector product evaluation.
     *
     * @param dst Vector where the computed evaluated vector-matrix product is stored.
     * @param src The solution vector at the current time.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    /**
     * This function sets the corresponding values on the fictional outer face if the face is
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

    CompressibleFlowScratchData<dim, number> &flow_scratch_data;

    const CompressibleFlowConvectiveKernels<dim, number> convective_terms;

    const CompressibleFlowViscousKernels<dim, number> viscous_terms;

    const MappingInfoType       &mapping_info_surface;
    const MappingInfoVectorType &mapping_info_cells;
    const MappingInfoVectorType &mapping_info_faces;

    dealii::FESystem<dim> fe_point_temp;
    const unsigned int    n_dofs_per_cell;

    mutable number inv_time_step = 0.;

    // Function, which describes the velocity of the unfitted object
    std::shared_ptr<dealii::Function<dim>> unfitted_object_velocity;
    // Inflow function for unfitted inflow boundary
    std::shared_ptr<dealii::Function<dim>> unfitted_inflow;
  };
} // namespace MeltPoolDG::Flow
