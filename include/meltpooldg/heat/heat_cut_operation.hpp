#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_cut_operator.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/heat_operation_base.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/cut_solution_transfer.hpp>
#include <meltpooldg/utilities/material_data.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <functional>
#include <memory>
#include <vector>


namespace MeltPoolDG::Heat
{
  template <int dim>
  class HeatCutOperation : public HeatOperationBase<dim>
  {
  private:
    using VectorType = HeatOperationBase<dim>::VectorType;

    const ScratchData<dim> &scratch_data;
    const HeatData<double> &heat_data;

    const TimeIterator<double> &time_iterator;

    const unsigned int temp_dof_idx;
    const unsigned int temp_hanging_nodes_dof_idx;
    const unsigned int temp_quad_idx;

    TimeIntegration::SolutionHistory<VectorType> solution_history;
    VectorType                                   volumetric_heat_source;

    // level set which defines the interface with its zero contour
    const unsigned int ls_dof_idx;
    const VectorType  &level_set;
    // The mesh classifier loops over all non-artificial cells and retrieves all level-set DoF
    // values to classify the cell. We use the @param mc_level_set vector that uses the matrix-based DoF layout.
    mutable VectorType                                        mc_level_set;
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier_old;

    CutUtil::SolutionTransferOperator<dim, double>                     cut_solution_transfer;
    std::function<void(const dealii::DoFHandler<dim> &)>               reinit_matrix_free;
    std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> reinit_vector;

    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<double>>
      mapping_info_surface;
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<double>>>>
      mapping_info_cells;

    NewtonRaphsonSolver<VectorType> newton;

    std::unique_ptr<HeatCutOperator<dim, double>> heat_operator;

    std::shared_ptr<
      Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorMatrixFree<dim, double>>>
      preconditioner_matrixfree;

    Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorMatrixFree<dim, double>>::
      PreconditionerObjectType preconditioner_used;

  public:
    HeatCutOperation(const ScratchData<dim>                     &scratch_data_in,
                     const HeatData<double>                     &heat_data_in,
                     const MaterialData<double>                 &material_data_in,
                     const Evaporation::EvaporationData<double> &evapor_data_in,
                     const TimeIterator<double>                 &time_iterator_in,
                     const unsigned int                          temp_dof_idx_in,
                     const unsigned int                          temp_hanging_nodes_dof_idx_in,
                     const unsigned int                          temp_quad_idx_in,
                     const bool                                  do_solidification_in,
                     const unsigned int                          ls_dof_idx_in,
                     const VectorType                           &level_set_in,
                     const unsigned int                          vel_dof_idx_in = 0,
                     const VectorType                           *velocity_in    = nullptr);

    void
    register_laser_intensity_function_and_direction(
      std::shared_ptr<const dealii::Function<dim, double>> laser_intensity_profile_in,
      const dealii::Tensor<1, dim, double>                &laser_direction_in);

    void
    register_reinit_matrix_free(
      const std::function<void(const dealii::DoFHandler<dim> &)> reinit_matrix_free_in);

  private:
    void
    classify_cells() const;

    void
    compute_intersected_quadrature();

    void
    adapt_to_new_interface_position();

  public:
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const override;

    void
    reinit() override;

    void
    set_initial_condition(const dealii::Function<dim> &initial_temperature) override;

    void
    distribute_constraints() override;

    void
    init_time_advance();

    void
    solve() override;

    /**
     * register vectors for adaptive mesh refinement solution transfer
     */
    void
    attach_vectors(std::vector<VectorType *> &vectors) override;

    /**
     * attach vectors for output
     */
    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const override;

    void
    attach_output_vectors_failed_step(GenericDataOut<dim> &data_out) const override;

    /*
     * getters
     */
    const VectorType &
    get_temperature() const override;

    VectorType &
    get_temperature() override;

    const VectorType &
    get_heat_source() const override;

    VectorType &
    get_heat_source() override;

  private:
    void
    setup_newton();
  };
} // namespace MeltPoolDG::Heat
