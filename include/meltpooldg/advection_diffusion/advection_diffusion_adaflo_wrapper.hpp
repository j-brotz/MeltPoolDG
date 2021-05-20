/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, December 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/advection_diffusion/advection_diffusion_operation_base.hpp>
#  include <meltpooldg/interface/scratch_data.hpp>
#  include <meltpooldg/interface/simulationbase.hpp>

#  include <adaflo/diagonal_preconditioner.h>
#  include <adaflo/level_set_okz_advance_concentration.h>
#  include <adaflo/level_set_okz_preconditioner.h>

namespace MeltPoolDG::AdvectionDiffusion
{
  template <int dim>
  class AdvectionDiffusionOperationAdaflo : public AdvectionDiffusionOperationBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    /**
     * Constructor.
     */
    AdvectionDiffusionOperationAdaflo(const ScratchData<dim> &scratch_data,
                                      int                     advec_diff_zero_dirichlet_dof_idx,
                                      int                     advec_diff_dirichlet_dof_idx,
                                      int                     advec_diff_quad_idx,
                                      int                     velocity_dof_idx,
                                      std::shared_ptr<SimulationBase<dim>> base_in,
                                      std::string operation_name = "advection_diffusion");

    void
    reinit() override;

    /**
     *  set initial solution of advected field
     */
    void
    set_initial_condition(const Function<dim> &initial_field_function,
                          const VectorType &   initial_velocity) override;

    /**
     * Solver time step
     */
    void
    solve(double dt, const VectorType &current_velocity) override;

    const LinearAlgebra::distributed::Vector<double> &
    get_advected_field() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_advected_field() override;

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    const LinearAlgebra::distributed::Vector<double> &
    get_advected_field_old() const;

    LinearAlgebra::distributed::Vector<double> &
    get_advected_field_old();

    const LinearAlgebra::distributed::Vector<double> &
    get_advected_field_old_old() const;

  private:
    void
    set_adaflo_parameters(const Parameters<double> &parameters,
                          int                       advec_diff_dof_idx,
                          int                       advec_diff_quad_idx,
                          int                       velocity_dof_idx);

    void
    set_velocity(const LinearAlgebra::distributed::Vector<double> &vec, bool initial_step = false);

    void
    initialize_vectors();

    const ScratchData<dim> &scratch_data;
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
    LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> bcs;
    /**
     * Adaflo parameters for the level set problem
     */
    LevelSetOKZSolverAdvanceConcentrationParameter adaflo_params;

    /**
     * Reference to the actual advection diffusion solver from adaflo
     */
    std::shared_ptr<LevelSetOKZSolverAdvanceConcentration<dim>> advec_diff_operation;

    /**
     *  maximum velocity --> set by adaflo
     */
    double global_max_velocity;
    /**
     *  Diagonal preconditioner @todo
     */
    DiagonalPreconditioner<double> preconditioner;
    const ConditionalOStream       pcout;
    /**
     *  dof idx for constraints with dirichlet values (relevant for dirichlet neq 0)
     */
    unsigned int dirichlet_dof_idx;
  };

} // namespace MeltPoolDG::AdvectionDiffusion

#endif
