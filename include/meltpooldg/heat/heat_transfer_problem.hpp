/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/heat/laser.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/problem_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/radiative_transport/rte_operation.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class HeatTransferProblem : public ProblemBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    VectorType velocity;
    VectorType level_set_as_heaviside;

    std::shared_ptr<TimeIterator<double>> time_iterator;
    DoFHandler<dim>                       dof_handler;
    DoFHandler<dim>                       dof_handler_velocity;
    DoFHandler<dim>                       dof_handler_level_set;

    AffineConstraints<double> temp_constraints;
    AffineConstraints<double> temp_hanging_nodes_constraints;
    AffineConstraints<double> velocity_hanging_nodes_constraints;
    AffineConstraints<double> level_set_hanging_nodes_constraints;
    // TODO: move to RTEOperation
    AffineConstraints<double> rte_constraints;
    AffineConstraints<double> rte_hanging_nodes_constraints;

    unsigned int temp_dof_idx;
    unsigned int temp_hanging_nodes_dof_idx;
    unsigned int temp_quad_idx;
    unsigned int velocity_dof_idx;
    unsigned int level_set_dof_idx;

    // TODO: move to RTEOperation
    unsigned int rte_dof_idx               = numbers::invalid_unsigned_int;
    unsigned int rte_hanging_nodes_dof_idx = numbers::invalid_unsigned_int;

    std::shared_ptr<ScratchData<dim>>           scratch_data;
    std::shared_ptr<HeatTransferOperation<dim>> heat_operation;
    std::shared_ptr<Material<double>>           material;
    std::shared_ptr<Postprocessor<dim>>         post_processor;

    std::shared_ptr<Function<dim>> velocity_field_function;
    std::shared_ptr<Function<dim>> heaviside_field_function;

    std::shared_ptr<Heat::LaserOperation<dim>>      laser_operation;
    std::shared_ptr<Heat::LaserHeatSourceBase<dim>> laser_heat_source_operation;

    std::shared_ptr<RadiativeTransport::RadiativeTransportOperation<dim>> rte_operation;

  public:
    HeatTransferProblem() = default;

    void
    run(std::shared_ptr<SimulationBase<dim>> base_in) final;

    std::string
    get_name() final;

  protected:
    void
    add_parameters(dealii::ParameterHandler &) final;

  private:
    struct
    {
      bool do_solidification = false;
    } problem_specific_parameters;
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize(std::shared_ptr<SimulationBase<dim>> base_in);

    void
    setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, bool do_reinit = true);

    void
    compute_field_vector(VectorType &vector, unsigned dof_idx, Function<dim> &field_function);

    /*
     *  perform output of results
     */
    void
    output_results(unsigned int                         time_step,
                   double                               current_time,
                   std::shared_ptr<SimulationBase<dim>> base_in);
    /*
     *  perform adaptive mesh refinement
     */
    void
    refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in);
  };
} // namespace MeltPoolDG::Heat
