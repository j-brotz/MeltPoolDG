#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/level_set/advection_DG_operator.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation_base.hpp>
#include <meltpooldg/level_set/advection_diffusion_operator.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>


namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class AdvectionDGOperation : public AdvectionDiffusionOperationBase<dim, number>
  {
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

  public:
    AdvectionDGOperation(
      const ScratchData<dim, dim, number>                          &scratch_data_in,
      const AdvectionDiffusionData<number>                         &advec_diff_data_in,
      const TimeIntegration::TimeIterator<number>                  &time_iterator,
      const unsigned int                                            advec_diff_dof_idx_in,
      const unsigned int                                            advec_diff_quad_idx_in,
      const std::shared_ptr<BoundaryConditionManager<dim, number>> &boundary_conditions_in);

    /**
     * Sets the initial conditions of the advection field based on a given analytical function.
     * The initial conditions are applied using a L_2 projection for each
     * element. This reduces oscillations for higher order elements.
     * @param initial_field_function Analytical initial condition
     */

    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) override;

    /**
     * Copies a given field to the initial conditions
     */
    void
    set_initial_condition(const VectorType &solution_level_set_in);


    void
    set_advection_velocity(const VectorType  &advection_velocity,
                           const unsigned int velocity_dof_idx) final;

    void
    set_advection_velocity_function(
      const std::shared_ptr<dealii::Function<dim, number>> &advection_velocity) final;

    void
    setup_constraints(
      ScratchData<dim, dim, number> & /*mutable_scratch_data*/,
      const PeriodicBoundaryConditions<dim> & /*pbc*/,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        & /*dirichlet_bc*/) final
    {
      // nothing to do for DG
    }

    /**
     * Allocates memory for the vectors based on the degrees of freedom of the DoFHandler.
     */
    void
    reinit() override;

    void
    init_time_advance() override;

    /**
     * Advances the solution in time by one time step according to the advection equation.
     */
    void
    solve(const bool do_finish_time_step = true) override;

    const VectorType &
    get_advected_field() const override;

    VectorType &
    get_advected_field() override;

    const VectorType &
    get_advected_field_old() const override;

    VectorType &
    get_advected_field_old() override;

    VectorType &
    get_user_rhs() override;

    const VectorType &
    get_user_rhs() const override;

    void
    attach_vectors(std::vector<VectorType *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

  private:
    const ScratchData<dim, dim, number> &scratch_data;

    const TimeIntegration::TimeIterator<number> &time_iterator;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int advec_diff_dof_idx  = 0;
    const unsigned int advec_diff_quad_idx = 0;

    std::shared_ptr<BoundaryConditionManager<dim, number>> boundary_conditions;

    std::shared_ptr<dealii::Function<dim>>                    advection_velocity_function = nullptr;
    const dealii::LinearAlgebra::distributed::Vector<number> *advection_velocity          = nullptr;
    unsigned int velocity_dof_idx = dealii::numbers::invalid_unsigned_int;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    VectorType rhs;
    VectorType user_rhs;

    std::shared_ptr<AdvectionDGOperator<dim, number>>            advection_DG_operator;
    std::shared_ptr<TimeIntegration::TimeIntegratorBase<number>> time_integrator;

    void
    create_operator();
  };
} // namespace MeltPoolDG::LevelSet
