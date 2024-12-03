#pragma once

#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_operator_base.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <unsigned int dim, typename number = double>
  class CompressibleFlowOperatorExplicit final : public CompressibleFlowOperatorBase<dim, number>
  {
    using VectorType = LinearAlgebra::distributed::Vector<number>;

  public:
    /**
     * Constructor.
     *
     * @param comp_flow_data_in Reference to the compressible flow data struct used.
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param comp_flow_dof_idx_in Index of the used dof handler in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     */
    CompressibleFlowOperatorExplicit(const CompressibleFlowData &comp_flow_data_in,
                                     const ScratchData<dim>     &scratch_data_in,
                                     unsigned int                comp_flow_dof_idx_in  = 0,
                                     unsigned int                comp_flow_quad_idx_in = 0);

    /**
     * Computes the value of the function f(y) for the compressible Navier-Stokes equations of the
     * form y' = f(y). From a discretization perspective, f(y) is given by f(y) = M^(-1) * F(y),
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
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func) const override;
  };
} // namespace MeltPoolDG::Flow