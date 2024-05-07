/* ---------------------------------------------------------------------
 *
 * Author: Johannes Resch, TUM, April 2024
 *
 * ---------------------------------------------------------------------*/
#pragma once



// MeltPoolDG
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#include <meltpooldg/reinitialization/reinitilization_DG_operator.hpp>
#include <meltpooldg/time_integration/time_integration_concretization.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>



namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  class ReinitializationDGOperation : public ReinitializationOperationBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    ReinitializationDGOperation(const ScratchData<dim>             &scratch_data_in,
                                const ReinitializationData<double> &reinit_data,
                                const TimeIterator<double>         &time_iterator,
                                const unsigned int                  reinit_dof_idx_in,
                                const unsigned int                  reinit_quad_idx_in,
                                const unsigned int                  ls_dof_idx_in);
    /**
     * Resizes the vectors to the right size of the underlying DoF handler
     */
    void
    reinit() override;

    /**
     * required from base class
     */
    void
    set_initial_condition([[maybe_unused]] const VectorType &solution_level_set_in) override{};

    /**
     * Sets the initial conditions of the advection field based on the analytical function
     * initial_field_function. The initial conditions are applied using a L_2 projection for each
     * element. This reduces oscillations for higher order elements.
     */
    void
    set_initial_condition(const Function<dim> &initial_field_function) override;

    void
    update_dof_idx(const unsigned int &reinit_dof_idx_in) override;

    void
    init_time_advance();

    /**
     * Performs one time step of the reinitialization process
     */
    void
    solve() override;

    double
    get_max_change_level_set() const final;

    /**
     * required from base class
     */
    const BlockVectorType &
    get_normal_vector() const override
    {
      AssertThrow(false, ExcNotImplemented());
    };


    const VectorType &
    get_level_set() const override;

    VectorType &
    get_level_set() override;

    /**
     * required from base class
     */
    BlockVectorType &
    get_normal_vector() override
    {
      AssertThrow(false, ExcNotImplemented());
    };

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const override;

    void
    prepare_reinitilization() override;

    /**
     * Computes the timestep size fullfilling the CFL condition
     */
    double
    compute_CFL_based_timestep() const override;



  private:
    const ScratchData<dim>            &scratch_data;
    const ReinitializationData<double> reinit_data;
    const TimeIterator<double>        &time_iterator;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
     *  multiple DoFHandlers, quadrature rules, etc.
     */
    mutable unsigned int reinit_dof_idx;
    const unsigned int   reinit_quad_idx;
    const unsigned int   ls_dof_idx;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    ReinitilizationDGOperator<dim> reinit_DG_operator;

    std::shared_ptr<TimeIntegrationBase<dim>> reinitilization_integration;

    VectorType rhs;

    // maximum change of the level set due to the current reinitialization step
    double max_change_level_set = std::numeric_limits<double>::max();
  };
} // namespace MeltPoolDG::LevelSet
