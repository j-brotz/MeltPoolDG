#pragma once

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/pseudo_rte_operator.hpp>
#include <meltpooldg/radiative_transport/rte_operator.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

#include <memory>

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

    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_hanging_nodes_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    VectorType rhs;

    std::unique_ptr<OperatorBase<dim, double>> rte_operator;
    std::unique_ptr<OperatorBase<dim, double>> pseudo_rte_operator;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    TimeSteppingData<double> pseudo_time_stepping;
    TimeIterator<double>     pseudo_time_iterator;

    Tensor<1, dim, double> laser_direction;

  public:
    RadiativeTransportOperation(const ScratchData<dim>               &scratch_data_in,
                                const RadiativeTransportData<double> &rte_data_in,
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
                      const PeriodicBoundaryConditions<dim>  &pbc,
                      const unsigned int                      rte_dof_idx_in,
                      const unsigned int                      rte_dof_hanging_nodes_dof_idx_in,
                      const bool                              set_inhomogeneities);

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
    void
    pseudo_solve();

    /*
     * Preconditioner for the matrix-free (pseudo-time dependent) RTE operator
     */
    std::shared_ptr<Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>
      preconditioner_matrixfree;
    std::shared_ptr<Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>
      pseudo_preconditioner_matrixfree;
    /*
     * Cache for diagonal preconditioner matrix-free
     */
    std::shared_ptr<DiagonalMatrix<VectorType>> diag_preconditioner_matrixfree;
    std::shared_ptr<DiagonalMatrix<VectorType>> diag_pseudo_preconditioner_matrixfree;
    /*
     * Cache for trilinos preconditioner matrix-free
     */
    std::shared_ptr<TrilinosWrappers::PreconditionBase> trilinos_preconditioner_matrixfree;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> trilinos_pseudo_preconditioner_matrixfree;
  };
} // namespace MeltPoolDG::RadiativeTransport
