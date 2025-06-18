#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/base/function.h>
#  include <deal.II/base/quadrature.h>
#  include <deal.II/base/tensor.h>
#  include <deal.II/base/timer.h>
#  include <deal.II/base/vectorization.h>

#  include <deal.II/dofs/dof_handler.h>

#  include <deal.II/grid/tria.h>

#  include <deal.II/lac/affine_constraints.h>
#  include <deal.II/lac/generic_linear_algebra.h>
#  include <deal.II/lac/la_parallel_vector.h>

#  include <meltpooldg/core/base_data.hpp>
#  include <meltpooldg/core/material.hpp>
#  include <meltpooldg/core/scratch_data.hpp>
#  include <meltpooldg/core/simulation_base.hpp>
#  include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#  include <meltpooldg/flow/incompressible_flow_operation_base.hpp>
#  include <meltpooldg/phase_change/evaporation_data.hpp>
#  include <meltpooldg/post_processing/generic_data_out.hpp>
#  include <meltpooldg/time_integration/time_iterator.hpp>
#  include <meltpooldg/time_integration/time_stepping_data.hpp>

#  include <adaflo/navier_stokes.h>
#  include <adaflo/parameters.h>

#  include <functional>
#  include <memory>
#  include <string>
#  include <vector>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Wrapper class integrating the Adaflo Navier-Stokes solver into the MeltPoolDG framework.
   *
   * This class serves as an adapter between the MeltPoolDG framework and the Adaflo library's
   * Navier-Stokes solver. It enables incompressible fluid flow simulations in a multiphysics
   * context of melt pool modeling with evaporation and phase change phenomena.
   *
   * `AdafloWrapper` derives from `IncompressibleFlowOperationBase` and provides all the necessary
   * interfaces to handle velocity and pressure fields, solver initialization, time stepping, and
   * postprocessing.
   */
  template <int dim, typename number>
  class AdafloWrapper : public IncompressibleFlowOperationBase<dim, number>
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * Initializes the Adaflo wrapper class.
     *
     * @param scratch_data Container containing mapping-, finite-element-, and quadrature-related
     * data
     * @param simulation_case Pointer to the base class for managing a simulation case.
     * @param adaflo_wrapper_params Struct containing all relevant wrapper parameters.
     * @param time_iterator Class for managing a time stepping routine.
     * @param do_evaporative_mass_flux Boolean variable indicating whether evaporative mass flux is
     * considered.
     */
    AdafloWrapper(ScratchData<dim, dim, number>                               &scratch_data,
                  const std::shared_ptr<const SimulationCaseBase<dim, number>> simulation_case,
                  const AdafloWrapperParameters<number>       &adaflo_wrapper_params,
                  const TimeIntegration::TimeIterator<number> &time_iterator,
                  const bool                                   do_evaporative_mass_flux);

    /**
     * @brief Set the initial velocity field for the Navier-Stokes solver.
     *
     * This function initializes the velocity field of the Adaflo Navier-Stokes solver using a
     * provided analytic or user-defined velocity function. The velocity is interpolated onto the
     * velocity DoFHandler and stored in the solution vector. Further, it applies hanging node
     * constraints to the velocity and pressure blocks.
     *
     * @param initial_field_function_velocity Function for the initial velocity field.
     *
     * @note This function modifies the 'current', 'old', and 'old-old' solution vectors to ensure
     * consistent initial values across multiple time levels for time-stepping schemes.
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function_velocity) override;

    /**
     * @brief Reinitialization of the flow solver.
     *
     * This function sets up degrees of freedom (DoFs), applies hanging node constraints,
     * and prepares internal data structures for the velocity and pressure fields.
     *
     * @note This method is typically called once before time stepping or solution initialization
     * to ensure all necessary data structures are correctly sized and constrained.
     */
    // TODO: Give this function a reasonable name
    void
    reinit_1();

    /**
     * @brief Reinitialization function to set up matrix-free infrastructure.
     *
     * This function initializes the matrix-free data structures used by the Adaflo Navier-Stokes
     * solver.
     *
     * @note This setup is required before performing matrix-free operator evaluations in Adaflo.
     */
    // TODO: Give this function a reasonable name
    void
    reinit_2();

    /**
     * @brief Reinitialization function to synchronize solution vectors for matrix-free execution.
     *
     * This function prepares and synchronizes velocity and pressure solution vectors across
     * multiple time levels (`solution`, `solution_old`, `solution_old_old`) for matrix-free
     * operation. This step is critical to avoid parallel inconsistencies and ensure correctness
     * when using matrix-free operations in MPI environments.
     */
    // TODO: Give this function a reasonable name
    void
    reinit_3();

    /**
     * @brief Initialize the time advancement procedure for the Navier-Stokes solver.
     *
     * This function sets up the time-stepping mechanism in the underlying Adaflo solver and
     * performs a consistency check to ensure that the time-stepping logic in Adaflo and the broader
     * MeltPoolDG framework are synchronized.
     *
     * @note This must be called before performing any time integration.
     */
    void
    init_time_advance() override;

    /**
     * @brief Solve the Navier-Stokes system for the current time step using the Adaflo solver.
     *
     * This function does the full nonlinear solution process for one time step of the
     * incompressible Navier-Stokes equations, including:
     *
     * - Initializing time advancement if not already done.
     * - Zeroing out user-defined right-hand side vectors (`user_rhs`) using velocity and pressure
     * constraints.
     * - Solving the nonlinear system via Adaflo's internal Newton solver.
     * - Recording iteration counts (both nonlinear and linear) for monitoring.
     * - Applying constraint distribution to update the solution with hanging node conditions.
     * - Computing and printing formatted norms of velocity and pressure fields for diagnostic
     * output.
     *
     * @note The function resets the `ready_for_time_advance` flag to `false`, enforcing that
     *       `init_time_advance()` must be called again before the next time step.
     */
    void
    solve() override;

    /**
     * @brief Assigns phase-dependent face densities for augmented Taylor-Hood elements.
     *
     * Set phase-dependent densities on faces for face integrals in augmented
     * Taylor-Hood elements with a pressure ansatz space containing elementwise
     * constant functions (element type FE_Q_DG0).
     *
     * This function calculates and assigns phase-dependent densities on faces for
     * use in face integrals of augmented Taylor-Hood elements.
     * The densities may depend on both the level-set field and the temperature
     * field.
     *
     * @param material Material object holding the phase-specific parameters necessary for density
     * calculation.
     * @param ls_as_heaviside DoF vector representing the indicator function.
     * @param ls_dof_idx DoF index of the indicator function within the matrix-free object.
     * @param temperature DoF vector representing the temperature field.
     * @param temp_dof_idx DoF index of the temperature function within the matrix-free object.
     */
    void
    set_face_average_density_augmented_taylor_hood(const Material<number> &material,
                                                   const VectorType       &ls_as_heaviside,
                                                   const unsigned int      ls_dof_idx,
                                                   const VectorType       *temperature  = nullptr,
                                                   unsigned int            temp_dof_idx = -1);

    /**
     * @brief Returns the current velocity vector (const).
     */
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity() const override;

    /**
     * @brief Returns the current velocity vector (non-const).
     */
    dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity() override;

    /**
     * @brief Returns the velocity vector from the previous time step (const).
     */
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity_old() const override;

    /**
     * @brief Returns the velocity vector from two time steps ago (const).
     */
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity_old_old() const;

    /**
     * @brief Returns the DoFHandler for the velocity field (const).
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler_velocity() const override;

    /**
     * @brief Returns the DoF index for the velocity field (const).
     */
    const unsigned int &
    get_dof_handler_idx_velocity() const override;

    /**
     * @brief Returns the DoF index for hanging node constraints in the velocity field (const).
     */
    const unsigned int &
    get_dof_handler_idx_hanging_nodes_velocity() const override;

    /**
     * @brief Returns the quadrature index used for velocity field (const).
     */
    const unsigned int &
    get_quad_idx_velocity() const override;

    /**
     * @brief Returns the quadrature index used for pressure field (const).
     */
    const unsigned int &
    get_quad_idx_pressure() const override;

    /**
     * @brief Returns the affine constraints for the velocity field (const).
     */
    const dealii::AffineConstraints<number> &
    get_constraints_velocity() const override;

    /**
     * @brief Returns a modifiable reference to the velocity field constraints (non-const).
     */
    dealii::AffineConstraints<number> &
    get_constraints_velocity() override;

    /**
     * @brief Returns the hanging node constraints for the velocity field (const).
     */
    const dealii::AffineConstraints<number> &
    get_hanging_node_constraints_velocity() const override;

    /**
     * @brief Returns the current pressure vector (const).
     */
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_pressure() const override;

    /**
     * @brief Returns a modifiable reference to the current pressure vector (non-const).
     */
    dealii::LinearAlgebra::distributed::Vector<number> &
    get_pressure() override;

    /**
     * @brief Returns the pressure vector from the previous time step (const).
     */
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_pressure_old() const;

    /**
     * @brief Returns the pressure vector from two time steps ago (const).
     */
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_pressure_old_old() const;

    /**
     * @brief Returns the DoFHandler for the pressure field (const).
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler_pressure() const override;

    /**
     * @brief Returns the DoF index for the pressure field (const).
     */
    const unsigned int &
    get_dof_handler_idx_pressure() const override;

    /**
     * @brief Returns the affine constraints for the pressure field (const).
     */
    const dealii::AffineConstraints<number> &
    get_constraints_pressure() const override;

    /**
     * @brief Returns a modifiable reference to the pressure field constraints (non-const).
     */
    dealii::AffineConstraints<number> &
    get_constraints_pressure() override;

    /**
     * @brief Returns the hanging node constraints for the pressure field (const).
     */
    const dealii::AffineConstraints<number> &
    get_hanging_node_constraints_pressure() const override;

    /**
     * @brief Sets the force right-hand side vector (momentum equation).
     *
     * @param vec The distributed vector to use as the force RHS.
     */
    void
    set_force_rhs(const dealii::LinearAlgebra::distributed::Vector<number> &vec) override;

    /**
     * @brief Sets the mass balance right-hand side vector (continuity equation).
     *
     * @param vec The distributed vector to use as the mass balance RHS.
     */
    void
    set_mass_balance_rhs(const dealii::LinearAlgebra::distributed::Vector<number> &vec) override;

    /**
     * @brief Sets a user-defined material law.
     *
     * This function allows the user to supply a custom constitutive model for computing the stress
     * tensor.
     *
     * @param my_user_defined_material A callable object returning a stress tensor from a given
     * strain-rate tensor.
     */
    void
    set_user_defined_material(std::function<dealii::Tensor<2, dim, dealii::VectorizedArray<number>>(
                                const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &,
                                const unsigned int,
                                const unsigned int,
                                const bool)> my_user_defined_material) override;

    /**
     * @brief Access (modifiable) to the fluid density at a given cell and quadrature point.
     *
     * @param cell The cell index.
     * @param q The quadrature point index.
     *
     * @return Reference to the density value.
     */
    dealii::VectorizedArray<number> &
    get_density(const unsigned int cell, const unsigned int q) override;

    /**
     * @brief Access (read-only) to the fluid density at a given cell and quadrature point.
     *
     * @param cell The cell index.
     * @param q The quadrature point index.
     *
     * @return Reference to the density value.
     */
    const dealii::VectorizedArray<number> &
    get_density(const unsigned int cell, const unsigned int q) const override;

    /**
     * @brief Access (modifiable) to the fluid viscosity at a given cell and quadrature point.
     *
     * @param cell The cell index.
     * @param q The quadrature point index.
     *
     * @return Reference to the viscosity value.
     */
    dealii::VectorizedArray<number> &
    get_viscosity(const unsigned int cell, const unsigned int q) override;

    /**
     * @brief Access (read-only) to the fluid viscosity at a given cell and quadrature point.
     *
     * @param cell The cell index.
     * @param q The quadrature point index.
     *
     * @return Reference to the viscosity value.
     */
    const dealii::VectorizedArray<number> &
    get_viscosity(const unsigned int cell, const unsigned int q) const override;

    /**
     * @brief Access (modifiable) to the damping coefficient at a given cell and quadrature point.
     *
     * @param cell The cell index.
     * @param q The quadrature point index.
     *
     * @return Reference to the damping coefficient.
     */
    dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q) override;

    /**
     * @brief Access (read-only) to the damping coefficient at a given cell and quadrature point.
     *
     * @param cell The cell index.
     * @param q The quadrature point index.
     *
     * @return Reference to the damping coefficient.
     */
    const dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q) const override;

    /**
     * @brief Attaches velocity solution vectors to a given container.
     *
     * This function appends pointers to the current, previous, and second-previous velocity
     * solution vectors to the provided vector.
     *
     * @param vectors A vector of pointers to distributed velocity vectors.
     *                The function will append three vectors: current, old, and old-old.
     */
    void
    attach_vectors_u(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    /**
     * @brief Attaches pressure solution vectors to a given container.
     *
     * This function appends pointers to the current, previous, and second-previous pressure
     * solution vectors to the provided vector.
     *
     * @param vectors A vector of pointers to distributed pressure vectors.
     *                he function will append three vectors: current, old, and old-old.
     */
    void
    attach_vectors_p(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    /**
     * @brief Apply constraints to the velocity and pressure solution vectors.
     *
     * This function distributes both the hanging node constraints and the general
     * constraints (e.g., Dirichlet boundary conditions) to the velocity and pressure
     * solution vectors for the current and previous two time steps.
     *
     * Specifically, it ensures that all solution vectors (velocity and pressure at
     * time levels n, n-1, and n-2) satisfy the imposed constraints.
     *
     * This step is typically required after solving the system but before using
     * the results (e.g., output or further calculations), ensuring consistency
     * and correctness in the constrained DoF space.
     */
    void
    distribute_constraints() override;

    /**
     * @brief Attach solution and auxiliary vectors to the output for visualization.
     *
     * This function registers a range of solution and auxiliary data vectors with
     * the given `GenericDataOut` object, enabling them to be written to visualization
     * files (e.g., VTU).
     *
     * The following fields may be attached:
     * - Velocity vector (current solution)
     * - Pressure scalar field
     * - Raw and projected force terms applied to velocity
     * - Raw and projected mass balance source terms
     * - Density and viscosity fields (if requested and available)
     *
     * @param data_out Data output object to which vectors will be attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    /**
     * @brief Attach residual and update vectors for failed Newton steps to output.
     *
     * This function adds data vectors relevant for diagnosing failed nonlinear
     * iterations (e.g., Newton solver failures). The following fields are added:
     * - Velocity update from the last nonlinear solve attempt
     * - Pressure update from the last nonlinear solve attempt
     * - Residual for the velocity field
     * - Residual for the pressure field
     *
     * All fields are forcibly added to the output regardless of `is_requested()`
     * to aid in debugging and post-processing.
     *
     * @param data_out Data output object to which diagnostic vectors will be attached.
     */
    void
    attach_output_vectors_failed_step(GenericDataOut<dim, number> &data_out) const override;

    /**
     * @brief Set the phase-averaged density value on a specific face of a cell.
     *
     * This function assigns a given density value to a face of the specified cell.
     *
     * @param cell The iterator to the cell whose face is being modified.
     * @param face The local face index.
     * @param density The averaged density value to assign to the specified face.
     */
    void
    set_face_average_density(const typename dealii::Triangulation<dim>::cell_iterator &cell,
                             const unsigned int                                        face,
                             const number                                              density);

    /**
     * @brief Returns a quadrature rule consisting of face center points.
     *
     * Constructs and caches a quadrature rule where each quadrature point
     * corresponds to the center of a face on a reference cell.
     *
     * @return A reference to a Quadrature object containing face center points.
     */
    const dealii::Quadrature<dim> &
    get_face_center_quad();

    /**
     * @brief Synchronizes the Adaflo time-stepping with the external time iterator.
     *
     * Ensures that the internal time-stepping state of Adaflo is consistent with the time iterator.
     * If discrepancies are detected, the Adaflo time-stepping is either restarted or advanced
     * appropriately to align with the current time and step count of the time iterator.
     */
    void
    synchronize_time_stepping();

  private:
    /// Scratch data object
    ScratchData<dim, dim, number> &scratch_data;

    /// Timer
    dealii::TimerOutput timer;

    /// Pointer to the Navier-Stokes solver from adaflo.
    std::unique_ptr<adaflo::NavierStokes<dim>> navier_stokes;

    /// Pointer to the face centered quadrature object
    std::unique_ptr<dealii::Quadrature<dim>> face_center_quad;

    /// Adaflo parameter struct
    const adaflo::FlowParameters &adaflo_params;

    /// Flag indicating whether evaporative mass flux is enabled in the simulation
    const bool do_evaporative_mass_flux;

    /// Time iterator object
    const TimeIntegration::TimeIterator<number> &time_iterator;

    /// DoF indices
    unsigned int dof_index_u;
    unsigned int dof_index_p;
    unsigned int dof_index_hanging_nodes_u;
    unsigned int dof_index_parameters;

    /// Quadrature indices
    unsigned int quad_index_u;
    unsigned int quad_index_p;

    /// DoFHandler for output of densities and viscosities
    dealii::DoFHandler<dim> dof_handler_parameters;

    /// Constraints for output of densities and viscosities
    dealii::AffineConstraints<number> constraints_parameters;

    /// Temporal vectors for output
    mutable VectorType force_rhs_velocity_projected;
    mutable VectorType mass_balance_source_term_projected;
    mutable VectorType density;
    mutable VectorType viscosity;

    /// Determine whether solution vectors are prepared for time advance
    bool ready_for_time_advance = false;

    /**
     * @brief Checks if the Adaflo time-stepping is synchronized with the time iterator.
     *
     * Compares the current and previous time, time steps, and step numbers between the internal
     * Adaflo time-stepping object and the `time_iterator`. The function uses a small
     * tolerance to account for floating-point rounding errors.
     *
     * @return true if all relevant time-stepping quantities match within a tight tolerance; false otherwise.
     *
     * This method is typically used to determine whether a call to synchronize_time_stepping() is
     * necessary.
     */
    bool
    time_stepping_synchronized();
  };
} // namespace MeltPoolDG::Flow
#endif
