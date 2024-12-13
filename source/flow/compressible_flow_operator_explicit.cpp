
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/flow/compressible_flow_operator_explicit.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>

namespace MeltPoolDG::Flow
{
  template <unsigned int dim, typename number>
  CompressibleFlowOperatorExplicit<dim, number>::CompressibleFlowOperatorExplicit(
    const CompressibleFlowData &comp_flow_data_in,
    const ScratchData<dim>     &scratch_data_in,
    unsigned int                comp_flow_dof_idx_in,
    unsigned int                comp_flow_quad_idx_in)
    : CompressibleFlowOperatorBase<dim, number>(comp_flow_data_in,
                                                scratch_data_in,
                                                comp_flow_dof_idx_in,
                                                comp_flow_quad_idx_in)
  {}

  template <unsigned int dim, typename number>
  void
  CompressibleFlowOperatorExplicit<dim, number>::apply_operator(
    number                                                 time,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    typedef std::function<void(const MatrixFree<dim, number> &,
                               LinearAlgebra::distributed::Vector<number>       &dst,
                               const LinearAlgebra::distributed::Vector<number> &src,
                               const std::pair<unsigned int, unsigned int> &)>
      local_applier_type;

    this->update_boundary_conditions(time);
    local_applier_type cell          = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_cell);
    local_applier_type face          = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_face);
    local_applier_type boundary_face = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_boundary_face);
    this->scratch_data.get_matrix_free().loop(cell, face, boundary_face, dst, src, true);

    local_applier_type inverse = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_inverse_mass_matrix);
    this->scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }


  template class CompressibleFlowOperatorExplicit<1, double>;
  template class CompressibleFlowOperatorExplicit<2, double>;
  template class CompressibleFlowOperatorExplicit<3, double>;
} // namespace MeltPoolDG::Flow