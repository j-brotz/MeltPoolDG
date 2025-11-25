#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/flow/compressible_flow_boundary_conditions.hpp>

#include <memory>
#include <utility>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Common interface class for compressible flow operation classes based on the type
   * erasure idiom.
   */
  template <int dim, typename number>
  class CfdDemFlowOperation
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Default constructor.
     */
    CfdDemFlowOperation() = default;

    /**
     * @brief Constructor, takes ownership of the passed unique pointer.
     *
     * @param operation Unique pointer to an operation object for a specific type of compressible
     * flow operation.
     */
    template <typename OperationType>
    explicit CfdDemFlowOperation(std::unique_ptr<OperationType> &&operation)
      : operation_pimpl(std::make_unique<OperationModel<OperationType>>(std::move(operation)))
    {}

    /**
     * @brief Solves the compressible Navier-Stokes equations for a single time step.
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
     * @brief Distribute the degrees of freedom to the passed dof handler object.
     */
    void
    distribute_dofs()
    {
      operation_pimpl->distribute_dofs();
    }

    /**
     * @brief Create the quadrature object and attach it to scratch data.
     */
    void
    create_quadrature()
    {
      operation_pimpl->create_quadrature();
    }

    /**
     * @brief Create the required constraints and attach it to scratch data.
     */
    void
    create_constraints()
    {
      operation_pimpl->create_constraints();
    }

    /**
     * @brief Set up the required internal data structures.
     *
     * After a call to this function the solve() function of the class can be utilized.
     */
    void
    reinit()
    {
      operation_pimpl->reinit();
    }

    /**
     * @brief Compute the maximum time step size.
     *
     * The maximum time step size arises from the convective and viscous time step limits.
     * It is optionally printed to the console.
     *
     * @param do_print If true, the time step limit is printed to the console.
     */
    number
    compute_time_step_size(const bool do_print = false) const
    {
      return operation_pimpl->compute_time_step_size(do_print);
    }

    /**
     * @brief Set the solution vector to the passed initial flow field state.
     *
     * @param function Initial condition of the flow field.
     */
    void
    set_initial_condition(const dealii::Function<dim> &function)
    {
      operation_pimpl->set_initial_condition(function);
    }

    /**
     * @brief Set the boundary conditions.
     *
     * @param simulation_case dealii::Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     *
     * @note The function simply passes the parameters to the corresponding operation function.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name)
    {
      operation_pimpl->set_boundary_conditions(simulation_case, operation_name);
    }

    void
    add_external_force(
      std::shared_ptr<AdditionalCellAndQuadOperation<dim, number>>         external_force_residuum,
      std::shared_ptr<AdditionalCellAndQuadOperationJacobian<dim, number>> external_force_jacobian =
        nullptr)
    {
      operation_pimpl->add_external_force(std::move(external_force_residuum),
                                          std::move(external_force_jacobian));
    }

    /**
     * @brief Get a constant reference to the DoFHandler used in the current operation.
     *
     * @return A constant reference to the dealii::DoFHandler object.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler() const
    {
      return operation_pimpl->get_dof_handler();
    }

    /**
     * @brief Get a constant reference to the solution vector.
     *
     * @return A constant reference to the solution vector.
     */
    const VectorType &
    get_solution() const
    {
      return operation_pimpl->get_solution();
    }

    /**
     * @brief Get a reference to the solution vector.
     *
     * @return A reference to the solution vector.
     */
    VectorType &
    get_solution()
    {
      return operation_pimpl->get_solution();
    }

    /**
     * @brief Get a reference to the solution vector in primitive variables
     * (pressure, velocity, temperature).
     *
     * @return A reference to the solution vector in primitive variables.
     */
    VectorType &
    get_solution_in_primitive_variables()
    {
      return operation_pimpl->get_solution_in_primitive_variables();
    }

    /**
     * @brief Attach the solution to the passed data out object.
     *
     * The solution is added in conservative variable formulation (density, momentum, energy
     * density) and primitive variable formulation (pressure, velocity, temperature).
     *
     * @param data_out Object to which the solution vector is attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const
    {
      operation_pimpl->attach_output_vectors(data_out);
    }

    /**
     * @brief Check if the compressible flow operation is initialized,
     *
     * Check that the current object holds a valid pointer to any compressible flow operation
     * object.
     *
     * @return True if object holds a valid preconditioner object.
     */
    bool
    is_initialized() const
    {
      return (operation_pimpl != nullptr);
    }

    /**
     * @brief Delete the compressible flow operation object stored in this class.
     */
    void
    clear()
    {
      operation_pimpl.reset(nullptr);
    }

  private:
    /**
     * @brief Abstract base class defining the interface for compressible flow operations.
     *
     * This struct defines the pure virtual interface used for implementing type erasure in the
     * CfdDemFlowOperation class. Concrete operation classes must implement this interface
     * to be used with CfdDemFlowOperation.
     */
    struct OperationConcept
    {
      virtual ~OperationConcept() = default;

      virtual void
      solve(number current_time, number time_step) = 0;

      virtual void
      distribute_dofs() = 0;

      virtual void
      create_quadrature() = 0;

      virtual void
      create_constraints() = 0;

      virtual void
      attach_output_vectors(GenericDataOut<dim, number> &data_out) const = 0;

      virtual void
      reinit() = 0;

      virtual void
      add_external_force(
        std::shared_ptr<AdditionalCellAndQuadOperation<dim, number>> external_force_residuum,
        std::shared_ptr<AdditionalCellAndQuadOperationJacobian<dim, number>>
          external_force_jacobian) = 0;

      virtual number
      compute_time_step_size(bool do_print = false) const = 0;

      virtual void
      set_initial_condition(const dealii::Function<dim> &function) = 0;

      virtual void
      set_boundary_conditions(
        const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
        const std::string                                      &operation_name) = 0;

      virtual const VectorType &
      get_solution() const = 0;

      virtual VectorType &
      get_solution() = 0;

      virtual VectorType &
      get_solution_in_primitive_variables() = 0;

      virtual const dealii::DoFHandler<dim> &
      get_dof_handler() const = 0;
    };

    /**
     * @brief Concrete implementation of OperationConcept using a specific compressible flow
     * operation type.
     *
     * This struct wraps an actual compressible flow operation object (templated type) and forwards
     * all interface calls defined in OperationConcept to it. This enables type-erased storage of
     * operation implementations in CfdDemFlowOperation.
     *
     * @tparam OperationType The concrete class type implementing the compressible flow operation
     * logic.
     */
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
      distribute_dofs() override
      {
        operation->distribute_dofs();
      }

      void
      create_quadrature() override
      {
        operation->create_quadrature();
      }

      virtual void
      create_constraints() override
      {
        operation->create_constraints();
      }

      void
      attach_output_vectors(GenericDataOut<dim, number> &data_out) const override
      {
        operation->attach_output_vectors(data_out);
      }

      void
      add_external_force(
        std::shared_ptr<AdditionalCellAndQuadOperation<dim, number>> external_force_residuum,
        std::shared_ptr<AdditionalCellAndQuadOperationJacobian<dim, number>>
          external_force_jacobian) override
      {
        operation->add_external_force(std::move(external_force_residuum),
                                      std::move(external_force_jacobian));
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
      set_initial_condition(const dealii::Function<dim> &function) override
      {
        operation->set_initial_condition(function);
      }

      void
      set_boundary_conditions(
        const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
        const std::string                                      &operation_name) override
      {
        operation->set_boundary_conditions(simulation_case, operation_name);
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

      VectorType &
      get_solution_in_primitive_variables() override
      {
        return operation->get_solution_in_primitive_variables();
      }

      const dealii::DoFHandler<dim> &
      get_dof_handler() const override
      {
        return operation->get_dof_handler();
      }

    private:
      /// Pointer to the concrete compressible flow operation object.
      std::unique_ptr<OperationType> operation;
    };

    /// Pointer to the actual compressible flow operation object to which the function calls
    /// are forwarded.
    std::unique_ptr<OperationConcept> operation_pimpl;
  };
} // namespace MeltPoolDG::Flow
