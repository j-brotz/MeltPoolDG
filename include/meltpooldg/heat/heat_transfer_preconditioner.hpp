/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, UIBK/TUM, April 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType>
  class MassMatrix
  {
  public:
    using VectorType  = LinearAlgebra::distributed::Vector<Number>;
    using vector_type = VectorType;

    MassMatrix(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
               const unsigned int                                  dof_idx,
               const unsigned int                                  quad_idx)
      : matrix_free(matrix_free)
      , dof_idx(dof_idx)
      , quad_idx(quad_idx)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      FECellIntegrator<dim, 1, Number> phi(matrix_free, dof_idx, quad_idx);

      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &, auto &dst, const auto &src, auto &range) {
          for (auto cell = range.first; cell < range.second; ++cell)
            {
              phi.reinit(cell);
              phi.gather_evaluate(src, true, false, false);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                phi.submit_value(phi.get_value(q), q);
              phi.integrate_scatter(true, false, dst);
            }
        },
        dst,
        src,
        true);
    }

  private:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
    const unsigned int                                  dof_idx;
    const unsigned int                                  quad_idx;
  };


  template <typename Operator>
  class InverseMassMatrix
  {
  public:
    using VectorType = typename Operator::vector_type;

    InverseMassMatrix(const Operator &op)
      : op(op)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      // invert mass matrix
      ReductionControl     reduction_control;
      SolverCG<VectorType> solver(reduction_control);
      solver.solve(op, dst, src, PreconditionIdentity());
    }

    const Operator &op;
  };
} // namespace MeltPoolDG::Heat
