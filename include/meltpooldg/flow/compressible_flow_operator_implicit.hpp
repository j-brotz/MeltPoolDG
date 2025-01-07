#pragma once

#include <meltpooldg/flow/compressible_flow_operator_implicit_base.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  class CompressibleFlowOperatorImplicit final
    : public CompressibleFlowOperatorImplicitBase<dim, number>
  {
  private:
    using Base                       = CompressibleFlowOperatorBase<dim, number>;
    using VectorType                 = typename Base::VectorType;
    using ConservedVariablesType     = typename Base::ConservedVariablesType;
    using ConservedVariablesGradType = typename Base::ConservedVariablesGradType;

  public:
    /**
     * Constructor, sets up the time integrator and sets the size of the solution_history object.
     *
     * @param compressible_flow_data_in Reference to the compressible flow data struct used.
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param solution_history_in Reference to the used solution_history object.
     * @param comp_flow_dof_idx_in Index of the used dof handler in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     */
    CompressibleFlowOperatorImplicit(
      const CompressibleFlowData                     &compressible_flow_data_in,
      const ScratchData<dim>                         &scratch_data_in,
      ::TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
      unsigned int                                    comp_flow_dof_idx_in  = 0,
      unsigned int                                    comp_flow_quad_idx_in = 0);

    /**
     * Perform a single time step. This effectively just calls time_integrator->perform_time_step().
     * Anything else including pre- and post-processing not performed by the time integrator needs
     * to be done ecternally.
     *
     * @param current_time Current simulation time.
     * @param time_step Current time step size.
     * @param pre_processing Preprocessing function passed to the time integrator (e.g. executed
     * before each Runge-Kutta step).
     * @param post_processing Postprocessing function passed to the time integrator (e.g. execcuted
     * after each Runge-Kutta step).
     */
    void
    advance_time_step(
      number                                                        current_time,
      number                                                        time_step,
      std::function<void(number, VectorType &, const VectorType &)> pre_processing  = {},
      std::function<void(number, VectorType &, const VectorType &)> post_processing = {}) override;

    /**
     * Reinitilaize the internal data structures, i.e. allocate memory for vectors storing temporary
     * solutions.
     */
    void
    reinit() override;

    /**
     * This function sets class member variables which are constant within a single time stage (e.g.
     * boundary conditions) and used in both the residual and jacobian calculation. A suitable time
     * integrator class is expected to call this functions before a call to compute_residual() and
     * compute_jacobian().
     *
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param old_solution_in (Fictional) solution at the previous time used to approximate the
     * temporal derivative by 1/dt*(U^(n)-U^(n-1)).
     * @param rhs_scaling_factor Factor used to scle the rhs, i.e. the final residual is given by
     * R=y'-a*f(y), where the variable 'a' is the passed factor.
     */
    void
    set_stage_constants(number            current_time,
                        number            time_step,
                        const VectorType &old_solution_in,
                        number            rhs_scaling_factor = 1.) const;

    /**
     * Compute the result of J*x, where J is the jacobian. The method on how to compute/approximate
     * the jacobian is defined by the user in the compressible flow data.
     *
     * @param src Source vector x with which the jacobian gets mulitplied.
     * @param dst Loation at which the result of J*x is stored.
     *
     * @throws Exception if the layout of the two given vectors @p src and @p dst are not identical.
     * @note This function assumes that the function set_stage_constants() has been called in advance.
     */
    void
    apply_jacobian(VectorType &dst, const VectorType &src) const;

    /**
     * Compute the negative residual, i.e. -(y'-F(y)) where y' is the temporal derivative of the
     * primary variables and F is the sum of all fluxes occuring in the compressible Navier-Stokes
     * equations (right-hand side).
     *
     * @param current_time Current physical time.
     * @param src Current solution vector used to compute the residual.
     * @param dst Vector in which the residual is stored.
     *
     * @throws Assert if the layout of the two given vectors @p src and @p dst are not identical.
     * @note This function assumes that the function set_stage_constants() has been called in advance.
     */
    void
    compute_residual(number current_time, const VectorType &src, VectorType &dst) const;

  private:
    /**
     * Compute the result of J*x, where J is the jacobian computed analytically.
     *
     * @param src Source vector x with which the jacobian gets mulitplied.
     * @param dst Loation at which the result of J*x is stored.
     *
     * @throws Exception if the layout of the two given vectors @p src and @p dst are not identical.
     */
    void
    apply_jacobian_analytic(const VectorType &src, VectorType &dst) const;

    /**
     * Compute the result of J*x, where J is the jacobian approximated by finite differences.
     *
     * @param src Source vector x with which the jacobian gets mulitplied.
     * @param dst Loation at which the result of J*x is stored.
     *
     * @throws Exception if the layout of the two given vectors @p src and @p dst are not identical.
     */
    void
    apply_jacobian_finite_differences(const VectorType &src, VectorType &dst) const;

    /**
     * The local cell applier computing the residual contribution of the cell.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param cell_range Cell range which is considered in the applier.
     */
    void
    local_cell_residual(const MatrixFree<dim, number>                    &matrix_free,
                        LinearAlgebra::distributed::Vector<number>       &dst,
                        const LinearAlgebra::distributed::Vector<number> &src,
                        const std::pair<unsigned int, unsigned int>      &cell_range) const;

    /**
     * The local cell applier computing the residual contribution of the inner faces.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_face_residual(const MatrixFree<dim, number>                    &matrix_free,
                        LinearAlgebra::distributed::Vector<number>       &dst,
                        const LinearAlgebra::distributed::Vector<number> &src,
                        const std::pair<unsigned int, unsigned int>      &face_range) const;

    /**
     * The local cell applier computing the residual contribution of the boundary faces.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_boundary_face_residual(const MatrixFree<dim, number>                    &matrix_free,
                                 LinearAlgebra::distributed::Vector<number>       &dst,
                                 const LinearAlgebra::distributed::Vector<number> &src,
                                 const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * Computes the contribution of the cells to the product of the Jacobian and the provided source
     * vector for a specified cell range. The current solution of the primary variables, required
     * for the Jacobian computation, is retrieved from the solution_history object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param cell_range The range of cells considered by the applier.
     */
    void
    local_cell_jacobian(const MatrixFree<dim, number>               &matrix_free,
                        VectorType                                  &dst,
                        const VectorType                            &src,
                        const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Computes the contribution of the inner faces to the product of the Jacobian and the provided
     * source vector for a specified face range. The current solution of the primary variables,
     * required for the Jacobian computation, is retrieved from the solution_history object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param face_range The range of faces considered by the applier.
     */
    void
    local_face_jacobian(const MatrixFree<dim, number>               &matrix_free,
                        VectorType                                  &dst,
                        const VectorType                            &src,
                        const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * Computes the contribution of the boundary faces to the product of the Jacobian and the
     * provided source vector for a specified face range. The current solution of the primary
     * variables, required for the Jacobian computation, is retrieved from the solution_history
     * object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param face_range The range of faces considered by the applier.
     */
    void
    local_boundary_face_jacobian(const MatrixFree<dim, number>                    &matrix_free,
                                 LinearAlgebra::distributed::Vector<number>       &dst,
                                 const LinearAlgebra::distributed::Vector<number> &src,
                                 const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * Implicit time integrator.
     */
    std::unique_ptr<TimeIntegratorBase<number, CompressibleFlowOperatorImplicit<dim, number>>>
      time_integrator;

    /**
     * When computing the residual this factor defines a scaling factor for the right-hand side. The
     * final residual is then given by R=y'-a*f(y), where 'a' is the factor defined by thi variable.
     */
    mutable number residual_rhs_scaling_factor = 1.;

    /**
     * This vector is used to compute the temporal derivative y' in the residual computation by
     * y'=(current_solution-old_solution)/dt. We do not take the old_solution directly from
     * solution_history as providing the option to pass a (fictional) old solution can have
     * performance advantages, e.g. in the bdf time integration. It can be set by the function
     * set_stage_constants().
     */
    mutable const VectorType *time_integrator_old_solution = nullptr;

    /**
     * This vector is used in the approximation of the jacobian by finite differences. Here the
     * vector stores the residual with a disturbed input.
     */
    mutable VectorType disturbed_residual;

    /**
     * Current time step size. This needs to be stored as this value is required by the local cell
     * appliers.
     */
    mutable number current_time_step;
  };
} // namespace MeltPoolDG::Flow