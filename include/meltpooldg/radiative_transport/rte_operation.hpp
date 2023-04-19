#pragma once

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/rte_operator.hpp>

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
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    VectorType intensity;
    VectorType rhs;

    std::shared_ptr<RadiativeTransportOperator<dim>> rte_operator;

  public:
    RadiativeTransportOperation(const ScratchData<dim> &              scratch_data_in,
                                const RadiativeTransportData<double> &rte_data_in,
                                const VectorType &                    heaviside_in,
                                const unsigned int                    rte_dof_idx_in,
                                const unsigned int                    rte_quad_idx_in,
                                const unsigned int                    hs_dof_idx_in);

    void
    set_initial_condition(const Function<dim> &initial_intensity);

    void
    reinit();

    void
    distribute_constraints();

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

  private:
    /*
     * Preconditioner for the matrix-free RTE operator
     */
    std::shared_ptr<Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>
      preconditioner_matrixfree;
    /*
     * Cache for diagonal preconditioner matrix-free
     */
    std::shared_ptr<DiagonalMatrix<VectorType>> diag_preconditioner_matrixfree;
    /*
     * Cache for trilinos preconditioner matrix-free
     */
    std::shared_ptr<TrilinosWrappers::PreconditionBase> trilinos_preconditioner_matrixfree;
  };
} // namespace MeltPoolDG::RadiativeTransport
