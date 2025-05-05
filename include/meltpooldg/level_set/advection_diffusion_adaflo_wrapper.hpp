#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/core/scratch_data.hpp>
#  include <meltpooldg/core/simulation_base.hpp>
#  include <meltpooldg/level_set/advection_diffusion_operation_base.hpp>
#  include <meltpooldg/post_processing/generic_data_out.hpp>
#  include <meltpooldg/time_integration/time_iterator.hpp>

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
      const VectorType                                &advection_velocity,
      const int                                        advec_diff_zero_dirichlet_dof_idx,
      const int                                        advec_diff_dirichlet_dof_idx,
      const int                                        advec_diff_quad_idx,
      const int                                        velocity_dof_idx,
      const TimeIntegration::TimeSteppingData<number> &time_stepping,
      const AdvectionDiffusionData<number>            &ls,
      const BoundaryConditionManager<dim, number>     &bc);

    void
    reinit() override;

    /**
     *  set initial solution of advected field
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) override;

    void
    init_time_advance() override;

    /**
     * Solve time step
     */
    void
    solve(const bool do_finish_time_step = true) override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() const override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old_old() const;

  private:
    void
    set_adaflo_parameters(const TimeIntegration::TimeSteppingData<number> &time_stepping,
                          const AdvectionDiffusionData<number>            &ls,
                          int                                              advec_diff_dof_idx,
                          int                                              advec_diff_quad_idx,
                          int                                              velocity_dof_idx);

    void
    set_velocity(bool initial_step = false);

    void
    initialize_vectors();

    const ScratchData<dim, dim, number>                      &scratch_data;
    const TimeIntegration::TimeIterator<number>              &time_iterator;
    const dealii::LinearAlgebra::distributed::Vector<number> &advection_velocity;
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
    VectorType velocity_vec;
    VectorType velocity_vec_old;
    VectorType velocity_vec_old_old;
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
    unsigned int dirichlet_dof_idx;
  };

} // namespace MeltPoolDG::LevelSet

#endif
