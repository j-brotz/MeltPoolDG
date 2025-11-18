#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/core/scratch_data.hpp>
#  include <meltpooldg/core/simulation_base.hpp>
#  include <meltpooldg/level_set/advection_diffusion_operation_base.hpp>
#  include <meltpooldg/post_processing/generic_data_out.hpp>
#  include <meltpooldg/time_integration/time_iterator.hpp>
#  include <meltpooldg/utilities/constraints.hpp>

#  include <adaflo/diagonal_preconditioner.h>
#  include <adaflo/level_set_okz_advance_concentration.h>
#  include <adaflo/level_set_okz_preconditioner.h>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class AdvectionDiffusionOperationAdaflo : public AdvectionDiffusionOperationBase<dim, number>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    /**
     * Constructor.
     */
    AdvectionDiffusionOperationAdaflo(
      const ScratchData<dim, dim, number>             &scratch_data,
      const TimeIntegration::TimeIterator<number>     &time_iterator,
      const int                                        advec_diff_zero_dirichlet_dof_idx,
      const int                                        advec_diff_dirichlet_dof_idx,
      const int                                        advec_diff_hanging_nodes_dof_idx,
      const int                                        advec_diff_quad_idx,
      const TimeIntegration::TimeSteppingData<number> &time_stepping,
      const AdvectionDiffusionData<number>            &ls,
      const BoundaryConditionManager<dim, number>     &bc);

    void
    reinit() final;

    /**
     *  set initial solution of advected field
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) final;

    void
    set_advection_velocity(const VectorType  &advection_velocity_in,
                           const unsigned int velocity_dof_idx_in) final;

    void
    set_advection_velocity_function(
      const std::shared_ptr<dealii::Function<dim>> &advection_velocity) final;

    void
    setup_constraints(
      ScratchData<dim, dim, number> &mutable_scratch_data,
      const PeriodicBoundaryConditions<dim> & /*pbc*/,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &dirichlet_bc_in) final;

    void
    init_time_advance() final;

    /**
     * Solve time step
     */
    void
    solve(const bool do_finish_time_step = true) final;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() const final;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() final;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() final;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() const final;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) final;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const final;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() const final;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() final;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old_old() const;

  private:
    void
    set_adaflo_parameters(const TimeIntegration::TimeSteppingData<number> &time_stepping,
                          const AdvectionDiffusionData<number>            &ls,
                          int                                              advec_diff_dof_idx,
                          int                                              advec_diff_quad_idx);

    void
    create_operator();

    void
    update_velocity_history(bool initial_step = false);

    void
    initialize_vectors();

    const ScratchData<dim, dim, number>         &scratch_data;
    const TimeIntegration::TimeIterator<number> &time_iterator;
    /**
     *  advected field
     */
    VectorType advected_field;
    VectorType advected_field_old;
    VectorType advected_field_old_old;
    /**
     *  vectors for the solution of the linear system
     */
    VectorType increment;
    VectorType rhs;

    /**
     *  velocity
     */
    const dealii::LinearAlgebra::distributed::Vector<number> *advection_velocity = nullptr;
    VectorType                                                velocity_vec;
    VectorType                                                velocity_vec_old;
    VectorType                                                velocity_vec_old_old;
    /**
     * Boundary conditions for the advection diffusion operation
     */
    adaflo::LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> bcs;
    /**
     * Adaflo parameters for the level set problem
     */
    adaflo::LevelSetOKZSolverAdvanceConcentrationParameter adaflo_params;

    /**
     * Reference to the actual advection diffusion solver from adaflo
     */
    std::shared_ptr<adaflo::LevelSetOKZSolverAdvanceConcentration<dim>> advec_diff_operation;

    /**
     *  maximum velocity --> set by adaflo
     */
    number global_max_velocity;
    /**
     *  Diagonal preconditioner @todo
     */
    adaflo::DiagonalPreconditioner<number> preconditioner;
    const ConditionalOStream               pcout;
    /**
     *  dof idx for constraints with dirichlet values (relevant for dirichlet neq 0)
     */
    const unsigned int dirichlet_dof_idx     = dealii::numbers::invalid_unsigned_int;
    const unsigned int hanging_nodes_dof_idx = dealii::numbers::invalid_unsigned_int;
  };

} // namespace MeltPoolDG::LevelSet

#endif
