#include <meltpooldg/flow/compressible_flow_operator_implicit_base.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  CompressibleFlowOperatorImplicitBase<dim, number>::CompressibleFlowOperatorImplicitBase(
    const CompressibleFlowData                     &compressible_flow_data_in,
    const ScratchData<dim>                         &scratch_data_in,
    ::TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
    unsigned int                                    comp_flow_dof_idx_in,
    unsigned int                                    comp_flow_quad_idx_in)
    : CompressibleFlowOperatorBase<dim, number>(compressible_flow_data_in,
                                                scratch_data_in,
                                                solution_history_in,
                                                comp_flow_dof_idx_in,
                                                comp_flow_quad_idx_in)
  {
    rs_div_c     = this->comp_flow_data.gamma - 1.0;
    lambda_div_c = this->comp_flow_data.thermal_conductivity /
                   this->comp_flow_data.specific_gas_constant * (this->comp_flow_data.gamma - 1.0);
  }

  template class CompressibleFlowOperatorImplicitBase<1, double>;
  template class CompressibleFlowOperatorImplicitBase<2, double>;
  template class CompressibleFlowOperatorImplicitBase<3, double>;
} // namespace MeltPoolDG::Flow