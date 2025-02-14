#pragma once

#include <deal.II/fe/fe_system.h>

#include <meltpooldg/cut/cut_util.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_operator_base.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <unsigned int dim, typename number = double>
  class CutCompressibleFlowOperator final : public CompressibleFlowOperatorBase<dim, number>
  {
    using VectorType            = LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

  public:
    using ConservedVariablesType     = Tensor<1, dim + 2, VectorizedArray<number>>;
    using ConservedVariablesGradType = Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>>;

    /**
     * Constructor.
     *
     * @param comp_flow_data_in Reference to the compressible flow data struct used.
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param mapping_info_surface_in dealii::NonMatching::MappingInfo object, provides the mapping
     * information computation and mapping data storage of the surface.
     * @param mapping_info_cells_in Vector of dealii::NonMatching::MappingInfo objects, provides
     * the mapping information computation and mapping data storage of the cells on the
     * inner subdomain and the outer subdomain, respectively.
     * @param mapping_info_faces_in Vector of dealii::NonMatching::MappingInfo objects, provides
     * the mapping information computation and mapping data storage of the faces on the
     * inner subdomain and the outer subdomain, respectively.
     * @param n_dofs_per_cell_in number of degrees of freedom per cell.
     * @param comp_flow_dof_idx_in Index of the used dof handler in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     */
    CutCompressibleFlowOperator(const CompressibleFlowData                     &comp_flow_data_in,
                                const ScratchData<dim>                         &scratch_data_in,
                                ::TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
                                const MappingInfoType       &mapping_info_surface_in,
                                const MappingInfoVectorType &mapping_info_cells_in,
                                const MappingInfoVectorType &mapping_info_faces_in,
                                const unsigned int           comp_flow_dof_idx_in  = 0,
                                const unsigned int           comp_flow_quad_idx_in = 0);

    /**
     * Set the inflow field function in the case of an unfitted inflow boundary.
     *
     * @param inflow_function Function which describes the inflow field at the unfitted inflow boundary.
     */
    void
    set_inflow_field_unfitted_boundary(std::shared_ptr<Function<dim>> &inflow_function) override;

    /**
     * Set the velocity function in the case of an unfitted (rigid) moving object.
     *
     * @param velocity_function Scalar function which describes the velocity of an unfitted (rigid) moving object
     */
    void
    set_unfitted_object_velocity(std::shared_ptr<Function<dim>> &velocity_function) override;

    /**
     * Local appliers for right-hand side evaluation.
     */
    void
    local_apply_cell(const MatrixFree<dim, number>                    &matrix_free,
                     LinearAlgebra::distributed::Vector<number>       &dst,
                     const LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void
    local_apply_face(const MatrixFree<dim, number>                    &matrix_free,
                     LinearAlgebra::distributed::Vector<number>       &dst,
                     const LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned int, unsigned int>      &face_range) const;

    void
    local_apply_boundary_face(const MatrixFree<dim, number>                    &matrix_free,
                              LinearAlgebra::distributed::Vector<number>       &dst,
                              const LinearAlgebra::distributed::Vector<number> &src,
                              const std::pair<unsigned int, unsigned int>      &face_range) const;

    /**
     * Local appliers for left-hand side matrix-vector product evaluation.
     */
    void
    local_apply_cell_lhs(const MatrixFree<dim, number>                    &matrix_free,
                         LinearAlgebra::distributed::Vector<number>       &dst,
                         const LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void
    local_apply_face_lhs(const MatrixFree<dim, number>                    &matrix_free,
                         LinearAlgebra::distributed::Vector<number>       &dst,
                         const LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int>      &face_range) const;

    void
    local_apply_boundary_face_lhs(const MatrixFree<dim, number>                    &matrix_free,
                                  LinearAlgebra::distributed::Vector<number>       &dst,
                                  const LinearAlgebra::distributed::Vector<number> &src,
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
               const VectorType &src) const override;

    /**
     * Function for the matrix-free matrix-vector product evaluation.
     *
     * @param dst Vector where the computed evaluated vector-matrix product is stored.
     * @param src The solution vector at the current time.
     */
    void
    vmult(LinearAlgebra::distributed::Vector<double>       &dst,
          const LinearAlgebra::distributed::Vector<double> &src) const override;

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
      const Point<dim, VectorizedArray<number>> &q_point,
      const ConservedVariablesType              &w_m,
      ConservedVariablesType                    &w_p,
      const ConservedVariablesGradType          &grad_w_m,
      ConservedVariablesGradType                &grad_w_p) const;

    const MappingInfoType       &mapping_info_surface;
    const MappingInfoVectorType &mapping_info_cells;
    const MappingInfoVectorType &mapping_info_faces;

    FESystem<dim>      fe_point_temp;
    const unsigned int n_dofs_per_cell;

    // Function, which describes the velocity of the unfitted object
    std::shared_ptr<Function<dim>> unfitted_object_velocity;
    // Inflow function for unfitted inflow boundary
    std::shared_ptr<Function<dim>> unfitted_inflow;
  };
} // namespace MeltPoolDG::Flow
