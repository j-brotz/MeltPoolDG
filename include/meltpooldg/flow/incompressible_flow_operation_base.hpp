#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Abstract base class for incompressible flow solvers.
   *
   * This interface defines the common operations and data access patterns required by
   * incompressible flow solvers in MeltPoolDG.
   */
  template <int dim, typename number>
  class IncompressibleFlowOperationBase
  {
  public:
    virtual ~IncompressibleFlowOperationBase() = default;
    /**
     * @brief Initialize data structures required before time stepping begins.
     */
    virtual void
    init_time_advance() = 0;

    /**
     * @brief Perform a single time step.
     */
    virtual void
    solve() = 0;

    /**
     * @brief Set the initial condition for the simulation using a given function.
     */
    virtual void
    set_initial_condition(const dealii::Function<dim> &) = 0;

    /**
     * @brief Get the current velocity vector (const).
     */
    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity() const = 0;

    /**
     * @brief Get the current velocity vector (non-const).
     */
    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity() = 0;

    /**
     * @brief Get the velocity from the previous time step (const).
     */
    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity_old() const = 0;

    /**
     * @brief Get the DoF handler for the velocity field.
     */
    virtual const dealii::DoFHandler<dim> &
    get_dof_handler_velocity() const = 0;

    /**
     * @brief Get the index for the velocity DoFHandler.
     */
    virtual const unsigned int &
    get_dof_handler_idx_velocity() const = 0;

    /**
     * @brief Get the index of the hanging node constraints for the velocity field.
     */
    virtual const unsigned int &
    get_dof_handler_idx_hanging_nodes_velocity() const = 0;


    /**
     * @brief Get the quadrature index used for velocity evaluation.
     */
    virtual const unsigned int &
    get_quad_idx_velocity() const = 0;

    /**
     * @brief Get the quadrature index used for pressure evaluation.
     */
    virtual const unsigned int &
    get_quad_idx_pressure() const = 0;

    /**
     * @brief Get velocity constraints (const).
     */
    virtual const dealii::AffineConstraints<number> &
    get_constraints_velocity() const = 0;

    /**
     * @brief Get reference to velocity constraints (non-const).
     */
    virtual dealii::AffineConstraints<number> &
    get_constraints_velocity() = 0;

    /**
     * @brief Get hanging node constraints for velocity.
     */
    virtual const dealii::AffineConstraints<number> &
    get_hanging_node_constraints_velocity() const = 0;

    /**
     * @brief Get the pressure vector (const).
     */
    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_pressure() const = 0;

    /**
     * @brief Get the pressure vector (non-const).
     */
    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_pressure() = 0;

    /**
     * @brief Get the DoF handler for the pressure field.
     */
    virtual const dealii::DoFHandler<dim> &
    get_dof_handler_pressure() const = 0;

    /**
     * @brief Get the internal index for the pressure DoFHandler.
     */
    virtual const unsigned int &
    get_dof_handler_idx_pressure() const = 0;

    /**
     * @brief Get pressure constraints (const).
     */
    virtual const dealii::AffineConstraints<number> &
    get_constraints_pressure() const = 0;

    /**
     * @brief Get pressure constraints (non-const).
     */
    virtual dealii::AffineConstraints<number> &
    get_constraints_pressure() = 0;

    /**
     * @brief Get hanging node constraints for pressure.
     */
    virtual const dealii::AffineConstraints<number> &
    get_hanging_node_constraints_pressure() const = 0;

    /**
     * @brief Set the external force right-hand side (e.g., gravity).
     *
     * @param vec Right-hand side vector for force contribution.
     */
    virtual void
    set_force_rhs(const dealii::LinearAlgebra::distributed::Vector<number> &vec) = 0;

    /**
     * @brief Set the right-hand side for the mass balance equation.
     *
     * @param vec Mass conservation right-hand side.
     */
    virtual void
    set_mass_balance_rhs(const dealii::LinearAlgebra::distributed::Vector<number> &vec) = 0;

    /**
     * @brief Set a user-defined constitutive material law.
     */
    virtual void
    set_user_defined_material(std::function<dealii::Tensor<2, dim, dealii::VectorizedArray<number>>(
                                const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &,
                                const unsigned int,
                                const unsigned int,
                                const bool)> my_user_defined_material) = 0;

    /**
     * @brief Get the density at a given cell and quadrature point (non-const).
     */
    virtual dealii::VectorizedArray<number> &
    get_density(const unsigned int cell, const unsigned int q) = 0;

    /**
     * @brief Get the density at a given cell and quadrature point (const).
     */
    virtual const dealii::VectorizedArray<number> &
    get_density(const unsigned int cell, const unsigned int q) const = 0;

    /**
     * @brief Get the viscosity at a given cell and quadrature point (non-const).
     */
    virtual dealii::VectorizedArray<number> &
    get_viscosity(const unsigned int cell, const unsigned int q) = 0;

    /**
     * @brief Get the viscosity at a given cell and quadrature point (const).
     */
    virtual const dealii::VectorizedArray<number> &
    get_viscosity(const unsigned int cell, const unsigned int q) const = 0;

    /**
     * @brief Get the damping at a given cell and quadrature point (non-const).
     */
    virtual dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q) = 0;

    /**
     * @brief Get the damping at a given cell and quadrature point (const).
     */
    virtual const dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q) const = 0;

    /**
     * @brief Attach the velocity vectors.
     */
    virtual void
    attach_vectors_u(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;

    /**
     * @brief Attach the pressure vectors.
     */
    virtual void
    attach_vectors_p(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;

    /**
     * @brief Distribute the constraints to velocity and pressure fields.
     */
    virtual void
    distribute_constraints() = 0;

    /**
     * @brief Attach velocity and pressure fields to a `GenericDataOut` object for visualization.
     *
     * @param data_out Output interface object.
     */
    virtual void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const = 0;

    /**
     * @brief Attach velocity and pressure vectors to output when a time step fails.
     *
     * @param data_out Output interface object.
     */
    virtual void
    attach_output_vectors_failed_step(GenericDataOut<dim, number> &data_out) const = 0;
  };
} // namespace MeltPoolDG::Flow
