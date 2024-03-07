#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/boundary_conditions.hpp>
#include <meltpooldg/interface/periodic_boundary_conditions.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/pseudo_rte_operation.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/radiative_transport/rte_operator.hpp>

#include <memory>
#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  /*
   * TODO
   */
  template <int dim>
  class RadiativeTransportOperation
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const ScratchData<dim> &scratch_data;

    const RadiativeTransportData<double> rte_data;

    const Tensor<1, dim, double> laser_direction;

    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_hanging_nodes_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    VectorType intensity;
    VectorType rhs;

    std::unique_ptr<RadiativeTransportOperator<dim, double>> rte_operator;

    std::unique_ptr<PseudoRTEOperation<dim>> pseudo_rte_operation;

  public:
    RadiativeTransportOperation(const ScratchData<dim>               &scratch_data_in,
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
    distribute_constraints();

    void
    setup_constraints(ScratchData<dim>                       &scratch_data,
                      const DirichletBoundaryConditions<dim> &bc_data,
                      const PeriodicBoundaryConditions<dim>  &pbc);

    void
    solve();

    const LinearAlgebra::distributed::Vector<double> &
    get_intensity() const;

    LinearAlgebra::distributed::Vector<double> &
    get_intensity();

    /**
     * register vectors for adaptive mesh refinement
     */
    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

    /**
     * attach vectors for output
     */
    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    void
    compute_heat_source(VectorType        &heat_source,
                        const unsigned int heat_source_dof_idx,
                        const bool         zero_out = true) const;

  private:
    std::shared_ptr<Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>
                                                        preconditioner_matrixfree;
    std::shared_ptr<DiagonalMatrix<VectorType>>         diag_preconditioner_matrixfree;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> trilinos_preconditioner_matrixfree;
  };
} // namespace MeltPoolDG::RadiativeTransport
