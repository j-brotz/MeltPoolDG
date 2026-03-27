#pragma once

#include <meltpooldg/compressible_flow/convective_kernels.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/compressible_flow/viscous_kernels.hpp>
#include <meltpooldg/compressible_flow/dg_operator_base.hpp>
#include <meltpooldg/time_integration/implicit_explicit_integrator.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Operator for the matrix-free evaluation of a compressible single-phase flow cutDG
   * formulation for implicit-explicit time integration.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   * @tparam is_viscous Indicates whether the flow is viscous.
   */
  template <int dim, typename number, bool is_viscous = true>
  class DGCompressibleFlowOperatorImplicitExplicit final
    : public DGCompressibleFlowOperatorBase<dim, number>
  {
  public:
    using VectorType             = dealii::LinearAlgebra::distributed::Vector<number>;
    using ConservedVariablesType = CompressibleFlow::ConservedVariablesType<dim, number>;
    using ConservedVariablesGradType =
      CompressibleFlow::ConservedVariablesGradientType<dim, number>;

    /**
     * @brief Constructor.
     *
     * @param flow_scratch_data Reference to flow scratch data object.
     */
    explicit DGCompressibleFlowOperatorImplicitExplicit(
      CompressibleFlowScratchData<dim, number> &flow_scratch_data);

    void
    add_external_force(
      std::shared_ptr<ExternalFlowForce<dim, number>>         external_force_residuum,
      std::shared_ptr<ExternalFlowForceJacobian<dim, number>> external_force_jacobian) override;

    /**
     * @brief Compute the matrix representation of the Jacobian.
     *
     * @param sparse_matrix Sparse matrix in which the resulting matrix representation is stored.
     */
    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &sparse_matrix) const;

    /**
     * @brief Compute the inverse elements of the diagonal of the Jacobian.
     *
     * @param diagonal Vector in which the inverse elements of the diagonal are stored.
     */
    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const;

    /**
     * @brief Reinitialize the internal data structures.
     *
     * The reinitialization includes setting a new required size for the solution history object
     * according to the demands of the used time integrator.
     */
    void
    reinit() override;

    /**
     * @brief Advances solver by a single time step.
     *
     * This function performs a single implicit-explicit time step of size @p time_step starting
     * from the solution at time @p time.
     *
     * @note The function does not take care about updating the solution history object or similar
     * operations which are not directly related to the integration. It **only** advances the
     * solution by a single time step starting from the current solution in the solution history
     * object of the @ref flow_scratch_data object.
     */
    void
    advance_time_step(number time, number time_step) override;

    /**
     * @brief Local cell operations at the given quadrature point for computing the Jacobian.
     *
     * @param delta_phi Cell integrator for the change in the primary variables. Quadrature point
     * distributions are added to this integrator.
     * @param phi Cell integrator for the primary variables.
     * @param q_index Quadrature point index.
     */
    void
    local_cell_jacobian_kernel(
      FECellIntegrator<dim, dim + 2, number>                      &delta_phi,
      const FECellIntegrator<dim, dim + 2, number>                &phi,
      unsigned int                                                 q_index,
      std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cell_iterators) const;

    /**
     * @brief Local face operations at the given quadrature point for computing the Jacobian.
     *
     * @param delta_phi_m Face integrator for the change in the primary variables on the inner face.
     * Quadrature point distributions are added to this integrator.
     * @param delta_phi_p Face integrator for the change in the primary variables on the outer face.
     * Quadrature point distributions are added to this integrator.
     * @param phi_m Cell integrator for the primary variables on the inner face.
     * @param phi_p Cell integrator for the primary variables on the outer face.
     * @param q_index Quadrature point index.
     */
    void
    local_face_jacobian_kernel(FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
                               FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_p,
                               const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
                               const FEFaceIntegrator<dim, dim + 2, number> &phi_p,
                               unsigned                                      q_index) const;

    /**
     * @brief Local boundary face operations at the given quadrature point for computing the Jacobian.
     *
     * @param delta_phi_m Face integrator for the change in the primary variables on the inner face.
     * Quadrature point distributions are added to this integrator.
     * @param phi_m Cell integrator for the primary variables on the inner face.
     * @param q_index Quadrature point index.
     */
    void
    local_boundary_face_jacobian_kernel(FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
                                        const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
                                        unsigned q_index) const;

    /**
     * @brief This function sets class member variables which are constant within a single time stage
     * (e.g. boundary conditions) and used in both the residual and jacobian calculation.
     *
     * A suitable time integrator class is expected to call this functions before a call to
     * compute_residual() and compute_jacobian().
     *
     * @param current_time Current physical time.
     * @param time_step Current time step size.
     * @param intermediate_explicit_solution_in Intermediate solution obtained by the explicit
     * time step.
     * @param rhs_scaling_factor Factor used to scale the rhs, i.e. the final residual is given by
     * R=y'-a*f(y), where the variable 'a' is the passed factor.
     */
    void
    set_stage_constants(number            current_time,
                        number            time_step,
                        const VectorType &intermediate_explicit_solution_in,
                        number            rhs_scaling_factor = 1.) const;

    /**
     * @brief Perform the explicit step to get an intermediate solution.
     *
     * @param current_time Current physical time.
     * @param time_step Current time step size.
     * @param dst Destination vector.
     * @param src Current solution vector used for the computations.
     * @param zero_dst_vec Flag whether the vector dst will be set to zero inside the loop.
     */
    void
    perform_explicit_stage(number                                         current_time,
                           number                                         time_step,
                           VectorType                                    &dst,
                           const VectorType                              &src,
                           bool                                           zero_dst_vec,
                           const std::function<void(unsigned, unsigned)> &post) const;

    /**
     * @brief Compute the result of J*x, where J is the Jacobian.
     *
     * The method on how to compute/approximate the jacobian is defined by the user in the
     * compressible flow data.
     *
     * @param src Source vector x with which the jacobian gets multiplied.
     * @param dst Location at which the result of J*x is stored.
     *
     * @throws Exception if the layout of the two given vectors @p src and @p dst are not identical.
     * @note This function assumes that the function set_stage_constants() has been called in advance.
     */
    void
    apply_jacobian(number time, number time_step, VectorType &dst, const VectorType &src) const;

    /**
     * @brief Compute the negative residual, i.e. -(y'-F(y)) where y' is the temporal derivative of
     * the primary variables and F is the sum of all fluxes occurring in the compressible
     * Navier-Stokes equations (right-hand side).
     *
     * @param current_time Current physical time.
     * @param src Current solution vector used to compute the residual.
     * @param dst Vector in which the residual is stored.
     *
     * @throws Assert if the layout of the two given vectors @p src and @p dst are not identical.
     * @note This function assumes that the function set_stage_constants() has been called in advance.
     */
    void
    compute_residual(number            current_time,
                     number            time_step,
                     const VectorType &src,
                     VectorType       &dst,
                     const VectorType &explicit_solution) const;

  private:
    /// Pointer to an intermediate explicit solution vector.
    mutable const VectorType *intermediate_explicit_solution = nullptr;

    /// This vector is used in the approximation of the Jacobian by finite differences. Here the
    /// vector stores the residual with a disturbed input.
    mutable VectorType disturbed_residual;

    /// Current time step size. This needs to be stored as this value is required by the local cell
    /// appliers.
    mutable number current_time_increment;

    /// Scratch data for compressible flows
    CompressibleFlowScratchData<dim, number> &flow_scratch_data;

    /// Time integrator class used for the time integration.
    TimeIntegration::ImplicitExplicitIntegrator<dim, number> time_integrator;

    /// Object for the convective term evaluations
    CompressibleFlowConvectiveKernels<dim, number> convective_terms;

    /// Object for the viscous term evaluations
    CompressibleFlowViscousKernels<dim, number> viscous_terms;

    /// This set of pointers may hold a list of external fluid force contributions to the explicitly
    /// treated part of the PDE (e.g., gravity or user-defined source terms)
    std::vector<std::shared_ptr<ExternalFlowForce<dim, number>>> external_forces_explicit_rhs;

    /// This set of pointers may hold a list of external fluid force contributions to the residuum
    /// (e.g., gravity or user-defined source terms)
    std::vector<std::shared_ptr<ExternalFlowForce<dim, number>>> external_forces_implicit_residual;

    /// This set of pointers may hold a list of external fluid force contributions to the jacobian
    /// (e.g., gravity or user-defined source terms)
    std::vector<std::shared_ptr<ExternalFlowForceJacobian<dim, number>>>
      external_forces_implicit_jacobian;

    /**
     * @brief The local cell applier for computing the intermediate explicit stage.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param cell_range Cell range which is considered in the applier.
     * */
    void
    local_cell_explicit_stage(const dealii::MatrixFree<dim, number>                    &matrix_free,
                              dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                              const dealii::LinearAlgebra::distributed::Vector<number> &src,
                              const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief The local face applier for computing the intermediate explicit stage.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_face_explicit_stage(const dealii::MatrixFree<dim, number>                    &matrix_free,
                              dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                              const dealii::LinearAlgebra::distributed::Vector<number> &src,
                              const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief The local boundary face applier for computing the intermediate explicit stage.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_boundary_face_explicit_stage(
      const dealii::MatrixFree<dim, number>                    &matrix_free,
      dealii::LinearAlgebra::distributed::Vector<number>       &dst,
      const dealii::LinearAlgebra::distributed::Vector<number> &src,
      const std::pair<unsigned int, unsigned int>              &face_range) const;

    /**
     * @brief The local cell applier computing the residual contribution of the cell.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param cell_range Cell range which is considered in the applier.
     */
    void
    local_cell_residual(const dealii::MatrixFree<dim, number>                    &matrix_free,
                        dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                        const dealii::LinearAlgebra::distributed::Vector<number> &src,
                        const std::pair<unsigned int, unsigned int>              &cell_range) const;

    /**
     * @brief The local face applier computing the residual contribution of the inner faces.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_face_residual(const dealii::MatrixFree<dim, number>                    &matrix_free,
                        dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                        const dealii::LinearAlgebra::distributed::Vector<number> &src,
                        const std::pair<unsigned int, unsigned int>              &face_range) const;

    /**
     * @brief The local cell boundary face computing the residual contribution of the boundary faces.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_boundary_face_residual(const dealii::MatrixFree<dim, number>              &matrix_free,
                                 dealii::LinearAlgebra::distributed::Vector<number> &dst,
                                 const dealii::LinearAlgebra::distributed::Vector<number> &src,
                                 const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Computes the contribution of the cells to the product of the Jacobian and the provided
     * source vector for a specified cell range.
     *
     * The current solution of the primary variables, required for the Jacobian computation, is
     * retrieved from the solution_history object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param cell_range The range of cells considered by the applier.
     */
    void
    local_cell_jacobian(const dealii::MatrixFree<dim, number>       &matrix_free,
                        VectorType                                  &dst,
                        const VectorType                            &src,
                        const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Computes the contribution of the inner faces to the product of the Jacobian and the
     * provided source vector for a specified face range.
     *
     * The current solution of the primary variables, required for the Jacobian computation, is
     * retrieved from the solution_history object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param face_range The range of faces considered by the applier.
     */
    void
    local_face_jacobian(const dealii::MatrixFree<dim, number>       &matrix_free,
                        VectorType                                  &dst,
                        const VectorType                            &src,
                        const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Computes the contribution of the boundary faces to the product of the Jacobian and the
     * provided source vector for a specified face range.
     *
     * The current solution of the primary variables, required for the Jacobian computation, is
     * retrieved from the solution_history object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param face_range The range of faces considered by the applier.
     */
    void
    local_boundary_face_jacobian(const dealii::MatrixFree<dim, number>              &matrix_free,
                                 dealii::LinearAlgebra::distributed::Vector<number> &dst,
                                 const dealii::LinearAlgebra::distributed::Vector<number> &src,
                                 const std::pair<unsigned int, unsigned int> &face_range) const;
  };
} // namespace MeltPoolDG::Flow
