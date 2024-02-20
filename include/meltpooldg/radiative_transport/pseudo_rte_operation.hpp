#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/radiative_transport/pseudo_rte_operator.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  /*
   * TODO
   */
  template <int dim>
  class PseudoRTEOperation
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const ScratchData<dim> &scratch_data;

    const RadiativeTransportData<double> &rte_data;

    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_hanging_nodes_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    VectorType rhs;

    std::unique_ptr<PseudoRTEOperator<dim, double>> pseudo_rte_operator;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    TimeIterator<double> pseudo_time_iterator;

    std::shared_ptr<Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>
                                                        preconditioner_matrixfree;
    std::shared_ptr<DiagonalMatrix<VectorType>>         diag_preconditioner_matrixfree;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> trilinos_preconditioner_matrixfree;

  public:
    PseudoRTEOperation(const ScratchData<dim>               &scratch_data_in,
                       const RadiativeTransportData<double> &rte_data_in,
                       const Tensor<1, dim, double>         &laser_direction_in,
                       const VectorType                     &heaviside_in,
                       const unsigned int                    rte_dof_idx_in,
                       const unsigned int                    rte_hanging_nodes_dof_idx_in,
                       const unsigned int                    rte_quad_idx_in,
                       const unsigned int                    hs_dof_idx_in);

    void
    reinit();

    void
    perform_pseudo_time_stepping();

    void
    solve();

    void
    set_intensity(const LinearAlgebra::distributed::Vector<double> &intensity_in);

    const LinearAlgebra::distributed::Vector<double> &
    get_predicted_intensity() const;
  };
} // namespace MeltPoolDG::RadiativeTransport