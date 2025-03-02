#pragma once

#include <deal.II/fe/fe_system.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>
#include <meltpooldg/flow/compressible_flow_eos_utils.hpp>

namespace MeltPoolDG::Multiphase
{
  template <unsigned int dim, typename number = double>
  class CompressibleMultiphaseOperator
  {
    using VectorType            = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

   using ConservedVariablesType     = Flow::CompressibleFlowTypes::ConservedVariablesType<dim, number>;
   using ConservedVariablesGradType = Flow::CompressibleFlowTypes::ConservedVariablesGradType<dim, number>;

  public:
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
    CompressibleMultiphaseOperator(MeltPoolDG::Flow::CompressibleFlowScratchData<dim, number> &flow_scratch_data,
                                  const MappingInfoType                    &mapping_info_surface_in,
                                  const MappingInfoVectorType              &mapping_info_cells_in,
                                  const MappingInfoVectorType              &mapping_info_faces_in);

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

    /**
     * Function which sets the inverse time step size.
     */
    void
    compute_inverse_time_step(const number &time_step);

  private:
    MeltPoolDG::Flow::CompressibleFlowScratchData<dim, number> &flow_scratch_data;

    const MeltPoolDG::Flow::CompressibleFlowConvectiveKernels<dim, number> convective_terms;

    const MeltPoolDG::Flow::CompressibleFlowViscousKernels<dim, number> viscous_terms;

    const MappingInfoType       &mapping_info_surface;
    const MappingInfoVectorType &mapping_info_cells;
    const MappingInfoVectorType &mapping_info_faces;
 
    dealii::FESystem<dim> fe_point_temp;
    const unsigned int    n_dofs_per_cell;

    mutable number velocity_interface = 0.;

    // Weighting factors for viscous phase interface flux
    double alpha_1;
    double alpha_2;

    number inv_time_step;

    // Function, which describes the velocity of the unfitted object
    std::shared_ptr<dealii::Function<dim>> unfitted_object_velocity;
    // Inflow function for unfitted inflow boundary
    std::shared_ptr<dealii::Function<dim>> unfitted_inflow;

     // TODO: Move into any util?
     FECellIntegrator<dim, dim + 2, number> create_cell_integrator(CutUtil::CellCategory category, unsigned int offset = 0) const
     {
       return FECellIntegrator<dim, dim + 2, number>(
           flow_scratch_data.scratch_data.get_matrix_free(),
           flow_scratch_data.dof_idx,
           flow_scratch_data.quad_idx,
           offset,
           category);
     }

     // TODO: Move into any util?
     // Create fe_face_integrator lambda function
     FEFaceIntegrator<dim, dim + 2, number> create_face_integrator(bool is_inner_face, CutUtil::CellCategory category, unsigned int offset = 0) const
     {
       return FEFaceIntegrator<dim, dim + 2, number>(
           flow_scratch_data.scratch_data.get_matrix_free(),
           is_inner_face,
           flow_scratch_data.dof_idx,
           flow_scratch_data.quad_idx,
           offset,
           category);
      };
   };
} // namespace MeltPoolDG::Multiphase
