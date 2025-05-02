#pragma once

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/curvature_DG_operation.hpp>
#include <meltpooldg/level_set/normal_vector_DG_operation.hpp>
#include <meltpooldg/level_set/reinitialization_DG_operator.hpp>
#include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>



namespace MeltPoolDG::LevelSet
{

  template <int dim, typename number>
  class ReinitializationDGOperation : public ReinitializationOperationBase<dim, number>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    ReinitializationDGOperation(const ScratchData<dim, dim, number>         &scratch_data_in,
                                const ReinitializationData<number>          &reinit_data,
                                const TimeIntegration::TimeIterator<number> &time_iterator,
                                const unsigned int                           reinit_dof_idx_in,
                                const unsigned int                           reinit_quad_idx_in,
                                const unsigned int                           ls_dof_idx_in,
                                const NormalVectorData<number>              &normal_vec_data,
                                const CurvatureData<number>                 &curvature_data);
    /**
     * For advection reinit coupled problems the normal vector and curvature are computed a level
     * higher on the level set operation level. This is because the computation of the normal vector
     * and curvature is very expensive and should only be done once when needed.
     */
    ReinitializationDGOperation(
      const ScratchData<dim, dim, number>                           &scratch_data_in,
      const ReinitializationData<number>                            &reinit_data,
      const TimeIntegration::TimeIterator<number>                   &time_iterator,
      const unsigned int                                             reinit_dof_idx_in,
      const unsigned int                                             reinit_quad_idx_in,
      const unsigned int                                             ls_dof_idx_in,
      const std::shared_ptr<NormalVectorOperationBase<dim, number>> &normal_vector_operation_in,
      const std::shared_ptr<CurvatureDGOperation<dim, number>>      &curvature_operation_in,
      const bool                                                     is_coupled_in);

    /**
     * Resizes the vectors to the right size of the underlying DoF handler
     */
    void
    reinit() override;

    /**
     * Copies a given field
     */
    void
    set_initial_condition(const VectorType &solution_level_set_in) override;

    /**
     * Sets the initial conditions of the level set field based on the analytical function
     * @param initial_field_function. The initial conditions are applied using a L_2 projection for each
     * element. This reduces oscillations for higher order elements.
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) override;

    void
    update_dof_idx(const unsigned int &reinit_dof_idx_in) override;

    void
    init_time_advance();

    /**
     * Performs one time step of the reinitialization process
     */
    void
    solve() override;

    number
    get_max_change_level_set() const final;

    const BlockVectorType &
    get_normal_vector() const override;

    const VectorType &
    get_level_set() const override;

    VectorType &
    get_level_set() override;

    BlockVectorType &
    get_normal_vector() override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    /**
     * Computes the timestep size fullfilling the CFL condition
     */
    number
    compute_CFL_based_timestep() const override;

    void
    set_artificial_diffusitivity() override;

    VectorType *
    get_sign_indicator_function() override
    {
      return &reinit_DG_operator->get_sign_indicator_function();
    }

  private:
    const ScratchData<dim, dim, number>         &scratch_data;
    const ReinitializationData<number>           reinit_data;
    const TimeIntegration::TimeIterator<number> &time_iterator;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    mutable unsigned int reinit_dof_idx;
    const unsigned int   reinit_quad_idx;
    const unsigned int   ls_dof_idx;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::shared_ptr<ReinitilizationDGOperator<dim, number>> reinit_DG_operator;

    std::shared_ptr<TimeIntegration::TimeIntegratorBase<number>> reinitialization_integration;


    // maximum change of the level set due to the current reinitialization step
    number max_change_level_set = std::numeric_limits<number>::max();

    /*
     *   Computation of the normal vectors
     */
    std::shared_ptr<NormalVectorOperationBase<dim, number>> normal_vector_operation;

    /*
     *   Computation of the curvature
     */
    std::shared_ptr<CurvatureDGOperation<dim, number>> curvature_operation;

    const bool is_coupled = false;
  };
} // namespace MeltPoolDG::LevelSet
