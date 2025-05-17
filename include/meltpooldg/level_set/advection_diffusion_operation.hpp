#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/boundary_conditions.hpp>
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation_base.hpp>
#include <meltpooldg/level_set/advection_diffusion_operator.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/constraints.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class AdvectionDiffusionOperation : public AdvectionDiffusionOperationBase<dim, number>
  {
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

  public:
    /*
     *  All the necessary parameters are stored in this struct.
     */

    AdvectionDiffusionOperation(
      const ScratchData<dim, dim, number> &scratch_data_in,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                  &dirichlet_bc_in,
      const AdvectionDiffusionData<number>        &advec_diff_data_in,
      const TimeIntegration::TimeIterator<number> &time_iterator,
      const unsigned int                           advec_diff_dof_idx_in,
      const unsigned int                           advec_diff_hanging_nodes_dof_idx_in,
      const unsigned int                           advec_diff_quad_idx_in);

    /**
     * @brief Set the initial condition for the advected field.
     *
     * Interpolates @p initial_field_function into the current solution,
     * distributes constraints, and initializes the most recent "old" state.
     *
     * @param initial_field_function Function describing the initial state.
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) override;

    /**
     * @brief Provide the advection velocity from an existing vector.
     *
     * @param advection_velocity Vector with velocity DoFs (compatible partitioning).
     * @param velocity_dof_idx   DoFHandler index in @c ScratchData for this vector.
     *
     * @note Mutually exclusive with set_advection_velocity_function(). One of them
     *       must be called before @ref reinit() / @ref init_time_advance() / @ref solve().
     */
    virtual void
    set_advection_velocity(const VectorType  &advection_velocity,
                           const unsigned int velocity_dof_idx) final;

    /**
     * @brief Provide the advection velocity via a (possibly time-dependent) function.
     *
     * @param advection_velocity Shared pointer to a dealii::Function returning a vector field.
     *
     * @note Mutually exclusive with set_advection_velocity(). One of them
     *       must be called before @ref reinit() / @ref init_time_advance() / @ref solve().
     */
    virtual void
    set_advection_velocity_function(
      const std::shared_ptr<dealii::Function<dim, number>> &advection_velocity) final;

    /**
     * @brief Set inflow/outflow boundary data (matrix-free mode only).
     *
     * When provided, the operator enforces inflow constraints at faces where
     * \f$ \boldsymbol{n}\cdot\boldsymbol{u} \le 0 \f$ using the supplied
     * boundary functions. Outflow is left unconstrained.
     *
     * @param inflow_outflow_bc_ Map of boundary id to scalar boundary function.
     *
     * @note Only supported when the linear solver runs in matrix-free mode.
     */
    void
    set_inflow_outflow_bc(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        inflow_outflow_bc_);

    /**
     * @brief Build and merge Dirichlet, hanging-node, and periodic constraints
     * into an AffineConstraints object.
     *
     * Populates constraints for the selected DoFHandler indices on
     * @p mutable_scratch_data. Periodic constraints are merged into the Dirichlet
     * set when applicable.
     *
     * @param mutable_scratch_data ScratchData container to be modified.
     * @param periodic_bc          Periodic boundary condition descriptor.
     * @param dirichlet_bc_in      Dirichlet BC map (may override constructor map if time-dependent).
     */
    void
    setup_constraints(
      ScratchData<dim, dim, number>         &mutable_scratch_data,
      const PeriodicBoundaryConditions<dim> &periodic_bc,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &dirichlet_bc_in) final;

    /**
     * @brief Allocate and (re)initialize vectors and the underlying operator.
     *
     * Initializes solution history vectors, right-hand-side, and preconditioner state.
     * Must be called after velocity and constraints are set.
     */
    void
    reinit() override;

    /**
     * @brief Prepare a time step: commit history, update constraints/BCs, build RHS,
     * compute predictor.
     *
     * Sets up the step according to the current @c TimeIterator and the
     * configured predictor. May create inflow/outflow constraints (matrix-free).
     */
    void
    init_time_advance() override;

    /**
     * @brief Solve the linear system for the current time step.
     *
     * Dispatches to matrix-free or matrix-based path depending on configuration,
     * applies preconditioner, enforces constraints, prints diagnostics, and
     * optionally finalizes the time step.
     *
     * @param do_finish_time_step If true, performs post-processing and
     *                            advances the time integrator state.
     */
    void
    solve(const bool do_finish_time_step = true) override;

    /**
     * @brief Register internal solution vectors for adaptive mesh refinement.
     *
     * @param vectors Vector of pointers to which internal solution vectors are appended.
     */
    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    /**
     * @brief Attach output fields to a @c GenericDataOut object.
     *
     * Adds the current advected field and the user-supplied RHS for visualization/output.
     *
     * @param data_out Output aggregator.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    /// @brief Get the current advected field (const).
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() const override;

    /// @brief Get the current advected field (mutable).
    dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() override;

    /// @brief Get the most recent old advected field (const).
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() const override;

    /// @brief Get the most recent old advected field (mutable).
    dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() override;

    /// @brief Get a mutable reference to the user-supplied RHS vector.
    dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() override;

    /// @brief Get a const reference to the user-supplied RHS vector.
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() const override;

  private:
    /**
     * @brief Instantiate and configure the underlying operator and preconditioner.
     *
     * Requires a valid advection velocity (vector or function) to be set.
     */
    void
    create_operator();

    /**
     * @brief Build inflow constraints (matrix-free mode) and pass indices to the operator.
     *
     * Collects face-DoF indices where \f$ \boldsymbol{n}\cdot\boldsymbol{u} \le 0 \f$
     * and associates them with values from @ref inflow_outflow_bc.
     *
     * @note No-op if @ref inflow_outflow_bc is empty.
     */
    void
    create_inflow_outflow_constraints();

    /**
     * @brief Compute the predictor solution for the current time step and
     * commit time history.
     */
    void
    commit_solution_history_and_compute_predictor();

    /// Read-only access to FE, DoFHandlers, quadratures, MF data, timers, etc.
    const ScratchData<dim, dim, number> &scratch_data;

    /// Dirichlet BC map provided at construction.
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &dirichlet_bc;

    /**
     * @brief Concrete advection–diffusion operator.
     *
     * User-configured via @c AdvectionDiffusionData; created in @ref create_operator().
     */
    std::unique_ptr<AdvectionDiffusionOperator<dim, number>> advec_diff_operator;

    /// Time iterator reference
    const TimeIntegration::TimeIterator<number> &time_iterator;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int advec_diff_dof_idx               = 0;
    const unsigned int advec_diff_quad_idx              = 0;
    const unsigned int advec_diff_hanging_nodes_dof_idx = dealii::numbers::invalid_unsigned_int;
    /// Optional velocity function (alternative to @ref advection_velocity).
    std::shared_ptr<dealii::Function<dim, number>> advection_velocity_function = nullptr;

    /// Optional pointer to an externally provided velocity vector.
    const dealii::LinearAlgebra::distributed::Vector<number> *advection_velocity = nullptr;

    /// DoFHandler index corresponding to @ref advection_velocity.
    unsigned int velocity_dof_idx = dealii::numbers::invalid_unsigned_int;

    /// History of solution vectors used by the predictor and time integrator.
    TimeIntegration::SolutionHistory<VectorType> solution_history;

    /// Predictor used to extrapolate the solution for the current step.
    std::unique_ptr<Predictor<VectorType, number>> predictor;

    /// Extrapolated advected field from the predictor.
    VectorType solution_advected_field_extrapolated;

    /// Preconditioner wrapper (matrix-based or matrix-free).
    Preconditioner<dim, VectorType, number> preconditioner;

    /// System right-hand side
    VectorType rhs;

    /// User-supplied right-hand side (added to the system RHS).
    VectorType user_rhs;

    /// Whether we updated ghost values for the old solution in @ref init_time_advance().
    bool update_ghosts = false;

    /// Inflow/outflow boundary data: boundary id → Dirichlet value function.
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> inflow_outflow_bc;

    /**
     * @brief Indices/values used to enforce inflow constraints in matrix-free mode.
     *
     * The first vector holds local indices into the partitioner; the second
     * stores the corresponding boundary values.
     */
    std::pair<std::vector<unsigned int>, std::vector<number>> inflow_constraints_indices_and_values;
  };
} // namespace MeltPoolDG::LevelSet
