/**
 * @brief Common interface class for compressible flow operation classes based on the type
 * erasure idiom.
 */

#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/flow/compressible_flow_boundary_conditions.hpp>

#include <memory>
#include <utility>

namespace MeltPoolDG::Flow
{

  template <unsigned int dim, typename number>
  class CompressibleFlowOperation
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    CompressibleFlowOperation() = default;

    /**
     * Constructor, takes ownership of the passed unique pointer.
     *
     * @param operation Unique pointer to an operation object for a specific type of compressible
     * flow operation.
     */
    template <typename OperationType>
    explicit CompressibleFlowOperation(std::unique_ptr<OperationType> &&operation)
      : operation_pimpl(std::make_unique<OperationModel<OperationType>>(std::move(operation)))
    {}

    /**
     * Solves the compressible Navier-Stokes equations for a single time step.
     *
     * @param current_time Current time at t^n.
     * @param time_step Current time step size.
     */
    void
    solve(const number current_time, const number time_step)
    {
      operation_pimpl->solve(current_time, time_step);
    }

    /**
     * Distribute the degrees of freedom to the passed dof handler object.
     *
     * @param dof_handler Dof handler object ussed for the compressible flow solver.
     */
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const
    {
      operation_pimpl->distribute_dofs(dof_handler);
    }

    /**
     * Set up the required internal data structures. After a call to this function the solve()
     * function of the class can be utilized.
     */
    void
    reinit()
    {
      operation_pimpl->reinit();
    }

    /**
     * Compute the maximum time step size arising from the convective and viscous time step limits
     * and optionally print it to the console.
     *
     * @param do_print If true, the time step limit is printed to the console.
     */
    number
    compute_time_step_size(const bool do_print = false) const
    {
      return operation_pimpl->compute_time_step_size(do_print);
    }

    /**
     * Set the solution vector to the passed initial flow field state.
     *
     * @param function Initial condition of the flow field.
     */
    void
    set_initial_condition(const Function<dim> &function)
    {
      operation_pimpl->set_initial_condition(function);
    }

    /**
     * Set the boundary conditions.
     *
     * @param simulation_case Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     *
     * @note The function simply passes the parameters to the corresponding operation function.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim>> &simulation_case,
                            const std::string                              &operation_name)
    {
      operation_pimpl->set_boundary_conditions(simulation_case, operation_name);
    }

    /**
     * Set a body force, e.g. gravity, specified by the passed function.
     *
     * @param body_force_in Function specifying the body force.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_body_force(std::unique_ptr<Function<dim>> body_force_in)
    {
      operation_pimpl->set_body_force(std::move(body_force_in));
    }

    const dealii::DoFHandler<dim> &
    get_dof_handler() const
    {
      return operation_pimpl->get_dof_handler();
    }

    const VectorType &
    get_solution() const
    {
      return operation_pimpl->get_solution();
    }

    VectorType &
    get_solution()
    {
      return operation_pimpl->get_solution();
    }

    /**
     * Attach the solution to the passed data out object. The solution which are added are the
     * density, the momentum and the energy density.
     *
     * @param data_out Object to which the solution vector is attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim, double> &data_out) const
    {
      // check, if single-phase or two-phase case is considered
      const auto &dof_handler = operation_pimpl->get_dof_handler();
      const bool  two_phase   = dof_handler.get_fe_collection().n_components() / (dim + 2) == 2;

      std::vector<std::string> names;

      if (not two_phase)
        {
          names.emplace_back("density");
          for (unsigned int d = 0; d < dim; ++d)
            names.emplace_back("momentum");
          names.emplace_back("energy");
        }
      else
        {
          names.emplace_back("density_liquid");
          for (unsigned int d = 0; d < dim; ++d)
            names.emplace_back("momentum_liquid");
          names.emplace_back("energy_liquid");

          names.emplace_back("density_gas");
          for (unsigned int d = 0; d < dim; ++d)
            names.emplace_back("momentum_gas");
          names.emplace_back("energy_gas");
        }

      std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      for (unsigned int d = 0; d < dim; ++d)
        interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);

      // add entries for two-phase case
      if (two_phase)
        {
          interpretation.push_back(DataComponentInterpretation::component_is_scalar);
          for (unsigned int d = 0; d < dim; ++d)
            interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
          interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        }

      data_out.add_data_vector(operation_pimpl->get_dof_handler(),
                               operation_pimpl->get_solution(),
                               names,
                               interpretation);
    }

    /**
     * Check if the compressible flow operation is initialized, i.e. that the current object holds a
     * valid pointer to any compressible flow operation object.
     *
     * @return True if object holds a valid preconditoner object.
     */
    bool
    is_initialized() const
    {
      return (operation_pimpl != nullptr);
    }

    /**
     * Delete the compressible flow operation object stored in this class.
     */
    void
    clear()
    {
      operation_pimpl.reset(nullptr);
    }

  private:
    struct OperationConcept
    {
      virtual ~OperationConcept() = default;

      virtual void
      solve(number current_time, number time_step) = 0;

      virtual void
      distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const = 0;

      virtual void
      reinit() = 0;

      virtual number
      compute_time_step_size(bool do_print = false) const = 0;

      virtual void
      set_initial_condition(const Function<dim> &function) = 0;

      virtual void
      set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim>> &simulation_case,
                              const std::string                              &operation_name) = 0;

      virtual void
      set_body_force(std::unique_ptr<Function<dim>> body_force_in) = 0;

      virtual const VectorType &
      get_solution() const = 0;

      virtual VectorType &
      get_solution() = 0;

      virtual const dealii::DoFHandler<dim> &
      get_dof_handler() const = 0;
    };

    template <typename OperationType>
    struct OperationModel final : public OperationConcept
    {
    public:
      explicit OperationModel(std::unique_ptr<OperationType> &&operation_in)
        : operation(std::move(operation_in))
      {}

      void
      solve(const number current_time, const number time_step) override
      {
        operation->solve(current_time, time_step);
      }

      void
      distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const override
      {
        operation->distribute_dofs(dof_handler);
      }

      void
      reinit() override
      {
        operation->reinit();
      }

      number
      compute_time_step_size(const bool do_print = false) const override
      {
        return operation->compute_time_step_size(do_print);
      }

      void
      set_initial_condition(const Function<dim> &function) override
      {
        operation->set_initial_condition(function);
      }

      void
      set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim>> &simulation_case,
                              const std::string &operation_name) override
      {
        operation->set_boundary_conditions(simulation_case, operation_name);
      }

      void
      set_body_force(std::unique_ptr<Function<dim>> body_force_in) override
      {
        operation->set_body_force(std::move(body_force_in));
      }

      const VectorType &
      get_solution() const override
      {
        return operation->get_solution();
      }

      VectorType &
      get_solution() override
      {
        return operation->get_solution();
      }

      const dealii::DoFHandler<dim> &
      get_dof_handler() const override
      {
        return operation->get_dof_handler();
      }

    private:
      std::unique_ptr<OperationType> operation;
    };

    /**
     * Pointer to the actual compressible flow operation object to which the function calls are
     * forwarded.
     */
    std::unique_ptr<OperationConcept> operation_pimpl;
  };
} // namespace MeltPoolDG::Flow
