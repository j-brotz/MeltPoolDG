#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/pseudo_rte_operation.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/radiative_transport/rte_operator.hpp>

#include <memory>
#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
  /*
   * TODO
   */
  template <int dim, typename number>
  class RadiativeTransportOperation
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    const ScratchData<dim, dim, number> &scratch_data;

    const RadiativeTransportData<number> rte_data;

    const dealii::Tensor<1, dim, number> laser_direction;

    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_hanging_nodes_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    VectorType intensity;
    VectorType rhs;

    std::unique_ptr<RadiativeTransportOperator<dim, number>> rte_operator;

    std::unique_ptr<PseudoRTEOperation<dim, number>> pseudo_rte_operation;

  public:
    RadiativeTransportOperation(const ScratchData<dim, dim, number>  &scratch_data_in,
                                const RadiativeTransportData<number> &rte_data_in,
                                const dealii::Tensor<1, dim, number> &laser_direction_in,
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
    setup_constraints(
      ScratchData<dim, dim, number> &scratch_data,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> &bc_data,
      const PeriodicBoundaryConditions<dim>                                              &pbc);

    void
    solve();

    const VectorType &
    get_intensity() const;

    VectorType &
    get_intensity();

    /**
     * register vectors for adaptive mesh refinement
     */
    void
    attach_vectors(std::vector<VectorType *> &vectors);

    /**
     * attach vectors for output
     */
    void
    attach_output_vectors(MeltPoolDG::GenericDataOut<dim, number> &data_out) const;

    void
    compute_heat_source(VectorType        &heat_source,
                        const unsigned int heat_source_dof_idx,
                        const bool         zero_out = true) const;

  private:
    Preconditioner<dim, VectorType> preconditioner;
  };
} // namespace MeltPoolDG::RadiativeTransport
