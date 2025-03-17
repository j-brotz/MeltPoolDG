#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_operation_base.hpp>
#include <meltpooldg/heat/laser_operation.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/material.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>

#include "heat_transfer_case.hpp"


namespace MeltPoolDG::Heat
{
  template <int dim>
  class HeatTransferProblem
  {
  private:
    using CaseType        = HeatTransferCase<dim>;
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<double>;

    const std::unique_ptr<CaseType> simulation_case;

    VectorType velocity;
    VectorType level_set_as_heaviside;
    VectorType level_set;

    std::shared_ptr<TimeIterator<double>> time_iterator;
    dealii::DoFHandler<dim>               dof_handler;
    dealii::DoFHandler<dim>               dof_handler_velocity;
    dealii::DoFHandler<dim>               dof_handler_level_set;

    dealii::AffineConstraints<double> temp_constraints;
    dealii::AffineConstraints<double> temp_hanging_nodes_constraints;
    dealii::AffineConstraints<double> velocity_hanging_nodes_constraints;
    dealii::AffineConstraints<double> level_set_hanging_nodes_constraints;

    unsigned int temp_dof_idx;
    unsigned int temp_hanging_nodes_dof_idx;
    unsigned int temp_quad_idx;
    unsigned int velocity_dof_idx;
    unsigned int level_set_dof_idx;

    // optional DoFHandler index for the HeatCutOperation's continuous DoFs
    // default value is invalid so we don't accidentally use a different DoFHandler
    unsigned int temp_cont_dof_idx = -1;

    std::shared_ptr<ScratchData<dim>>               scratch_data;
    std::shared_ptr<HeatOperationBase<dim, double>> heat_operation;
    std::shared_ptr<Material<double>>               material;
    std::shared_ptr<Postprocessor<dim, double>>     post_processor;

    std::shared_ptr<dealii::Function<dim>> velocity_field_function;
    std::shared_ptr<dealii::Function<dim>> heaviside_field_function;
    std::shared_ptr<dealii::Function<dim>> level_set_field_function;

    std::shared_ptr<LaserOperation<dim, double>> laser_operation;

  public:
    HeatTransferProblem(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();

  private:
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize();

    void
    setup_dof_system();

    void
    compute_field_vector(VectorType            &vector,
                         unsigned               dof_idx,
                         dealii::Function<dim> &field_function);

    /*
     *  perform output of results
     */
    void
    output_results(const bool output_not_converged = false);
    /*
     * collect all relevant output data
     */
    void
    attach_output_vectors(GenericDataOut<dim, double> &data_out) const;
    /*
     *  perform adaptive mesh refinement
     */
    void
    refine_mesh();
  };
} // namespace MeltPoolDG::Heat
