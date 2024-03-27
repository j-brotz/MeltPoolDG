/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_operation_base.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operator.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  class AdvectionDiffusionOperation : public AdvectionDiffusionOperationBase<dim>
  {
  private:
    using VectorType       = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
    using SparseMatrixType = TrilinosWrappers::SparseMatrix;

  public:
    /*
     *  All the necessary parameters are stored in this struct.
     */

    AdvectionDiffusionOperation(const ScratchData<dim>               &scratch_data_in,
                                const AdvectionDiffusionData<double> &advec_diff_data_in,
                                const TimeIterator<double>           &time_iterator,
                                const VectorType                     &advection_velocity,
                                const unsigned int                    advec_diff_dof_idx_in,
                                const unsigned int advec_diff_hanging_nodes_dof_idx_in,
                                const unsigned int advec_diff_quad_idx_in,
                                const unsigned int velocity_dof_idx_in);

    /**
     * Provide a field function for the initial solution of the advected field
     */
    void
    set_initial_condition(const Function<dim> &initial_field_function) override;

    void
    reinit() override;

    void
    init_time_advance() override;

    void
    solve(const bool do_finish_time_step = true) override;

    void
    create_inflow_outflow_constraints();

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

    void
    set_inflow_outflow_bc(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> inflow_outflow_bc_);

  private:
    void
    create_operator(const VectorType &advection_velocity);

    const ScratchData<dim> &scratch_data;
    /*
     *  This pointer will point to your user-defined advection_diffusion operator.
     */
    std::unique_ptr<OperatorBase<dim, double>> advec_diff_operator;

    const TimeIterator<double>                       &time_iterator;
    const LinearAlgebra::distributed::Vector<double> &advection_velocity;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
     *  multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int advec_diff_dof_idx  = 0;
    const unsigned int advec_diff_quad_idx = 0;
    const unsigned int advec_diff_hanging_nodes_dof_idx;
    const unsigned int velocity_dof_idx = 0;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::unique_ptr<Predictor<VectorType, double>> predictor;

    /*
     *    This is the primary solution variable of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType solution_advected_field_extrapolated;
    /*
     * Preconditioner for the matrix-free advection diffusion operator
     */
    std::shared_ptr<Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>
      preconditioner_matrixfree;

    VectorType rhs;
    VectorType user_rhs;

    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> inflow_outflow_bc;

    std::pair<std::vector<unsigned int>, std::vector<double>> inflow_constraints_indices_and_values;
  };
} // namespace MeltPoolDG::LevelSet
