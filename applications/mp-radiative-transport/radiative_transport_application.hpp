#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_operation.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>
#include <utility>
#include <vector>

#include "radiative_transport_case.hpp"


namespace MeltPoolDG::RadiativeTransport
{
  template <int dim, typename number>
  class RadiativeTransportApplication
  {
  private:
    using CaseType   = RadiativeTransportCase<dim, number>;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    std::unique_ptr<CaseType> simulation_case;

    VectorType heaviside;
    VectorType heat_source;

    dealii::DoFHandler<dim> dof_handler;
    dealii::DoFHandler<dim> dof_handler_heaviside; // TODO remove and use above

    dealii::AffineConstraints<number> constraints_dirichlet;
    dealii::AffineConstraints<number> hanging_node_constraints;
    dealii::AffineConstraints<number> hanging_node_constraints_heaviside;

    unsigned int rte_dof_idx;
    unsigned int rte_hanging_nodes_dof_idx;
    unsigned int rte_quad_idx;
    unsigned int hs_dof_idx;

    std::shared_ptr<ScratchData<dim, dim, number>>         scratch_data;
    std::shared_ptr<TimeIntegration::TimeIterator<number>> time_iterator;

    std::shared_ptr<RadiativeTransportOperation<dim, number>> rad_trans_operation;

    std::shared_ptr<Postprocessor<dim, number>>          post_processor;
    std::unique_ptr<Profiling::ProfilingMonitor<number>> profiling_monitor;

    dealii::Tensor<1, dim, number> laser_direction;

  public:
    RadiativeTransportApplication(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();

  private:
    /*
     *  This function initializes the relevant scratch data for the computation of the radiative
     * transfer problem
     */
    void
    initialize();

    void
    setup_dof_system(const bool do_reinit = true);

    void
    compute_heaviside(dealii::Function<dim> &heaviside_func);

    void
    output_results(const unsigned int time_step, const number current_time);

    void
    refine_mesh();
  };
} // namespace MeltPoolDG::RadiativeTransport
