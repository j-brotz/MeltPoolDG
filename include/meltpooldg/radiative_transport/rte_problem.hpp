#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/problem_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/radiative_transport/rte_operation.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>
#include <string>
#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim>
  class RadiativeTransportProblem : public ProblemBase<dim>
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    VectorType heaviside;
    VectorType heat_source;

    DoFHandler<dim> dof_handler;
    DoFHandler<dim> dof_handler_heaviside; // TODO remove and use above

    AffineConstraints<double> constraints_dirichlet;
    AffineConstraints<double> hanging_node_constraints;
    AffineConstraints<double> hanging_node_constraints_heaviside;

    unsigned int rte_dof_idx;
    unsigned int rte_hanging_nodes_dof_idx;
    unsigned int rte_quad_idx;
    unsigned int hs_dof_idx;

    std::shared_ptr<ScratchData<dim>>     scratch_data;
    std::shared_ptr<TimeIterator<double>> time_iterator;

    std::shared_ptr<RadiativeTransportOperation<dim>> rte_operation;

    std::shared_ptr<Postprocessor<dim>>                  post_processor;
    std::unique_ptr<Profiling::ProfilingMonitor<double>> profiling_monitor;

    Tensor<1, dim, double> laser_direction;
    std::vector<double>    laser_direction_input_prm;

  public:
    RadiativeTransportProblem() = default;

    void
    run(std::shared_ptr<SimulationBase<dim>> base_in) final;

    std::string
    get_name() final;

  protected:
    void
    add_parameters(dealii::ParameterHandler &) final;

    void
    check_input_parameters();

  private:
    /*
     *  This function initializes the relevant scratch data
     *  for the computation of the RTE problem
     */
    void
    initialize(std::shared_ptr<SimulationBase<dim>> base_in);

    void
    setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, const bool do_reinit = true);

    void
    compute_heaviside(Function<dim> &heaviside_func);

    /*
     *  perform output of results
     */
    void
    output_results(const unsigned int                   time_step,
                   const double                         current_time,
                   std::shared_ptr<SimulationBase<dim>> base_in);

    /*
     *  perform mesh refinement
     */
    void
    refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in);
  };
} // namespace MeltPoolDG::RadiativeTransport
