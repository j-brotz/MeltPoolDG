#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/boundary_conditions.hpp>
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation_base.hpp>
#include <meltpooldg/level_set/advection_diffusion_operator.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

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
      const ScratchData<dim, dim, number>                                        &scratch_data_in,
      const std::map<dealii::types::boundary_id, std::shared_ptr<Function<dim>>> &dirichlet_bc_in,
      const AdvectionDiffusionData<number> &advec_diff_data_in,
      const TimeIterator<number>           &time_iterator,
      const VectorType                     &advection_velocity,
      const unsigned int                    advec_diff_dof_idx_in,
      const unsigned int                    advec_diff_hanging_nodes_dof_idx_in,
      const unsigned int                    advec_diff_quad_idx_in,
      const unsigned int                    velocity_dof_idx_in);

    /**
     * Provide a field function for the initial solution of the advected field
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) override;

    void
    reinit() override;

    void
    init_time_advance() override;

    void
    solve(const bool do_finish_time_step = true) override;

    void
    create_inflow_outflow_constraints();

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() const override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    void
    set_inflow_outflow_bc(const std::map<dealii::types::boundary_id, std::shared_ptr<Function<dim>>>
                            inflow_outflow_bc_);

  private:
    void
    create_operator(const VectorType &advection_velocity);

    const ScratchData<dim, dim, number>                                        &scratch_data;
    const std::map<dealii::types::boundary_id, std::shared_ptr<Function<dim>>> &dirichlet_bc;
    /*
     *  This pointer will point to your user-defined advection_diffusion operator.
     */
    std::unique_ptr<AdvectionDiffusionOperator<dim, number>> advec_diff_operator;

    const TimeIterator<number>                               &time_iterator;
    const dealii::LinearAlgebra::distributed::Vector<number> &advection_velocity;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int advec_diff_dof_idx  = 0;
    const unsigned int advec_diff_quad_idx = 0;
    const unsigned int advec_diff_hanging_nodes_dof_idx;
    const unsigned int velocity_dof_idx = 0;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::unique_ptr<Predictor<VectorType, number>> predictor;

    /*
     *    This is the primary solution variable of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType solution_advected_field_extrapolated;

    Preconditioner<dim, VectorType> preconditioner;

    VectorType rhs;
    VectorType user_rhs;

    std::map<dealii::types::boundary_id, std::shared_ptr<Function<dim>>> inflow_outflow_bc;

    std::pair<std::vector<unsigned int>, std::vector<number>> inflow_constraints_indices_and_values;
  };
} // namespace MeltPoolDG::LevelSet
