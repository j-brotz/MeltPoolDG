/* ---------------------------------------------------------------------
 *
 * Author: Johannes Resch, TUM, April 2024
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/advection_diffusion/advection_DG_operator.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operation_base.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operator.hpp>
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>


namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  class AdvectionDGOperation : public AdvectionDiffusionOperationBase<dim>
  {
  private:
    using VectorType       = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
    using SparseMatrixType = TrilinosWrappers::SparseMatrix;

  public:
    AdvectionDGOperation(
      const ScratchData<dim>                                &scratch_data_in,
      const AdvectionDiffusionData<double>                  &advec_diff_data_in,
      const TimeIterator<double>                            &time_iterator,
      VectorType                                            &advection_velocity,
      const unsigned int                                     advec_diff_dof_idx_in,
      const unsigned int                                     advec_diff_quad_idx_in,
      const unsigned int                                     velocity_dof_idx_in,
      const std::shared_ptr<BoundaryConditionManager<dim>> &&boundary_conditions_in,
      std::shared_ptr<dealii::Function<dim>>               &&advection_field_in,
      bool const                                             enable_analytical_velocity_update_in);

    /**
     * Sets the initial conditions of the advection field based on a given analytical function.
     * The initial conditions are applied using a L_2 projection for each
     * element. This reduces oscillations for higher order elements.
     * @param initial_field_function Analytical initial condition
     */

    void
    set_initial_condition(const Function<dim> &initial_field_function) override;

    /**
     * Copies a given field to the initial conditions
     */
    void
    set_initial_condition(const VectorType &solution_level_set_in);

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

    const LinearAlgebra::distributed::Vector<double> &
    get_advected_field() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_advected_field() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_advected_field_old() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_advected_field_old() override;

    LinearAlgebra::distributed::Vector<double> &
    get_user_rhs() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_user_rhs() const override;

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const override;


  private:
    const ScratchData<dim> &scratch_data;

    const TimeIterator<double>                       &time_iterator;
    const LinearAlgebra::distributed::Vector<double> &advection_velocity;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
     *  multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int advec_diff_dof_idx  = 0;
    const unsigned int advec_diff_quad_idx = 0;
    const unsigned int velocity_dof_idx    = 0;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    VectorType rhs;
    VectorType user_rhs;

    AdvectionDGOperator<dim> advection_DG_operator;

    std::shared_ptr<TimeIntegratorBase<double, AdvectionDGOperator<dim>>> advection_integration;
  };
} // namespace MeltPoolDG::LevelSet
