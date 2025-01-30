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
  template <int dim>
  class RadiativeTransportProblem
  {
  private:
    using CaseType   = RadiativeTransportCase<dim>;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

    std::unique_ptr<CaseType> simulation_case;

    VectorType heaviside;
    VectorType heat_source;

    dealii::DoFHandler<dim> dof_handler;
    dealii::DoFHandler<dim> dof_handler_heaviside; // TODO remove and use above

    dealii::AffineConstraints<double> constraints_dirichlet;
    dealii::AffineConstraints<double> hanging_node_constraints;
    dealii::AffineConstraints<double> hanging_node_constraints_heaviside;

    unsigned int rte_dof_idx;
    unsigned int rte_hanging_nodes_dof_idx;
    unsigned int rte_quad_idx;
    unsigned int hs_dof_idx;

    std::shared_ptr<ScratchData<dim>>     scratch_data;
    std::shared_ptr<TimeIterator<double>> time_iterator;

    std::shared_ptr<RadiativeTransportOperation<dim>> rad_trans_operation;

    std::shared_ptr<Postprocessor<dim>>                  post_processor;
    std::unique_ptr<Profiling::ProfilingMonitor<double>> profiling_monitor;

    dealii::Tensor<1, dim, double> laser_direction;
    std::vector<double>            laser_direction_input_prm;

  public:
    RadiativeTransportProblem(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();

  private:
    void
    add_parameters(dealii::ParameterHandler &);

    void
    check_input_parameters();

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
    output_results(const unsigned int time_step, const double current_time);

    void
    refine_mesh();
  };
} // namespace MeltPoolDG::RadiativeTransport
