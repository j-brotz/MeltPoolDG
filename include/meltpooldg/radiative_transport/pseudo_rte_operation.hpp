#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/radiative_transport/pseudo_rte_operator.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>

namespace MeltPoolDG::RadiativeTransport
{
  /*
   * TODO
   */
  template <int dim, typename number>
  class PseudoRTEOperation
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    const ScratchData<dim, dim, number> &scratch_data;

    const RadiativeTransportData<number> &rte_data;

    const VectorType &heaviside;

    const unsigned int rte_dof_idx;
    const unsigned int rte_hanging_nodes_dof_idx;
    const unsigned int rte_quad_idx;
    const unsigned int hs_dof_idx;

    VectorType rhs;

    std::unique_ptr<PseudoRTEOperator<dim, number>> pseudo_rte_operator;

    TimeIntegration::SolutionHistory<VectorType, number> solution_history;

    TimeIterator<number> pseudo_time_iterator;

    Preconditioner<dim, VectorType, number> preconditioner;

  public:
    PseudoRTEOperation(const ScratchData<dim, dim, number>  &scratch_data_in,
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
    perform_pseudo_time_stepping();

    void
    solve();

    void
    set_intensity(const VectorType &intensity_in);

    const VectorType &
    get_predicted_intensity() const;
  };
} // namespace MeltPoolDG::RadiativeTransport
