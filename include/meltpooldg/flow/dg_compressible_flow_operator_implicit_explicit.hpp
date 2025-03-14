#pragma once

#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_base.hpp>
#include <meltpooldg/time_integration/implicit_explicit_integrator.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number, bool is_viscous = true>
  class DGCompressibleFlowOperatorImplicitExplicit final
    : public DGCompressibleFlowOperatorBase<number>
  {
    using VectorType             = dealii::LinearAlgebra::distributed::Vector<number>;
    using ConservedVariablesType = CompressibleFlowTypes::ConservedVariablesType<dim, number>;
    using ConservedVariablesGradType =
      CompressibleFlowTypes::ConservedVariablesGradType<dim, number>;

  public:
    /**
     * Constructor.
     *
     * @param flow_scratch_data Reference to flow scratch data object.
     */
    explicit DGCompressibleFlowOperatorImplicitExplicit(
      CompressibleFlowScratchData<dim, number> &flow_scratch_data);

    /**
     * Compute the matrix representation of the Jacobian.
     *
     * @param sparse_matrix Sparse matrix in which the resulting matrix representation is stored.
     */
    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &sparse_matrix) const;

    /**
     * Compute the inverse elements of the diagonal of the jacobian.
     *
     * @param diagonal Vector in which the inverse elements of the diagonal are stored.
     */
    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const;

    /**
     * Reinitilaize the internal data structures, i.e. allocate memory for vectors storing temporary
     * solutions.
     */
    void
    reinit() override;

    /**
     * Creates and returns an implicit-explicit time integrator object which is set up with the
     * current operator.
     *
     * @param time_integrator_data Reference to the time integrator data object.
     *
     * @return Unique pointer to a time integrator which is templated on the own operator type.
     *
     * @throws If the time integrator type in the time integrator data is not an implicit-explicit
     * time integrator.
     */
    std::unique_ptr<TimeIntegratorBase<number>>
    make_problem_specific_time_integrator(const TimeIntegratorData &time_integrator_data) override;

    /**
     * Local cell operations at the given quadrature point for computing the jacobian.
     *
     * @param delta_phi Cell integrator for the change in the primary variables. Quadrature point
     * distributions are added to this integrator.
     * @param phi Cell integrator for the primary varibales.
     * @param q_index Quadrature point index.
     */
    void
    local_cell_jacobian_kernel(FECellIntegrator<dim, dim + 2, number>       &delta_phi,
                               const FECellIntegrator<dim, dim + 2, number> &phi,
                               unsigned int                                  q_index) const;


    /**
     * Local face operations at the given quadrature point for computing the jacobian.
     *
     * @param delta_phi_m Face integrator for the change in the primary variables on the inner face.
     * Quadrature point distributions are added to this integrator.
     * @param delta_phi_p Face integrator for the change in the primary variables on the outer face.
     * Quadrature point distributions are added to this integrator.
     * @param phi_m Cell integrator for the primary varibales on the inner face.
     * @param phi_p Cell integrator for the primary varibales on the outer face.
     * @param q_index Quadrature point index.
     */
    void
    local_face_jacobian_kernel(FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
                               FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_p,
                               const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
                               const FEFaceIntegrator<dim, dim + 2, number> &phi_p,
                               unsigned                                      q_index) const;

    /**
     * Local biundary face operations at the given quadrature point for computing the jacobian.
     *
     * @param delta_phi_m Face integrator for the change in the primary variables on the inner face.
     * Quadrature point distributions are added to this integrator.
     * @param phi_m Cell integrator for the primary varibales on the inner face.
     * @param q_index Quadrature point index.
     */
    void
    local_boundary_face_jacobian_kernel(FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
                                        const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
                                        unsigned q_index) const;

    /**
     * Perform an initial guess for the solution of at the next time step. The guessed solution is
     * thereby given by the intermediate explicit solution computed during the explicit step of the
     * imex scheme.
     *
     * @param solution Vector in which the solution guess is stored.
     */
    void
    make_initial_guess(VectorType &solution) const;

    /**
     * This function sets class member variables which are constant within a single time stage
     * (e.g. boundary conditions) and used in both the residual and jacobian calculation. A
     * suitable time integrator class is expected to call this functions before a call to
     * compute_residual() and compute_jacobian().
     *
     * @param current_time Current physical time.
     * @param time_step Current time step size.
     * @param intermediate_explicit_solution_in Intermediate solution obtained by the explicit
     * time step.
     * @param rhs_scaling_factor Factor used to scle the rhs, i.e. the final residual is given by
     * R=y'-a*f(y), where the variable 'a' is the passed factor.
     */
    void
    set_stage_constants(number            current_time,
                        number            time_step,
                        const VectorType &intermediate_explicit_solution_in,
                        number            rhs_scaling_factor = 1.) const;

    /**
     * Perform the explicit step to get an intermediate solution.
     *
     * @param current_time
     * @param time_step
     * @param dst
     * @param src
     */
    void
    perform_explicit_stage(number            current_time,
                           number            time_step,
                           VectorType       &dst,
                           const VectorType &src,
                           bool              zero_dst_vec = false) const;

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
     * The local cell applier for computing the intermediate explicit stage.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param cell_range Cell range which is considered in the applier.
     * */
    void
    local_cell_explicit_stage(const MatrixFree<dim, number>                    &matrix_free,
                              LinearAlgebra::distributed::Vector<number>       &dst,
                              const LinearAlgebra::distributed::Vector<number> &src,
                              const std::pair<unsigned int, unsigned int>      &cell_range) const;

    /**
     * The local face applier for computing the intermediate explicit stage.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_face_explicit_stage(const MatrixFree<dim, number>                    &matrix_free,
                              LinearAlgebra::distributed::Vector<number>       &dst,
                              const LinearAlgebra::distributed::Vector<number> &src,
                              const std::pair<unsigned int, unsigned int>      &face_range) const;

    /**
     * The local boundary face applier for computing the intermediate explicit stage.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_boundary_face_explicit_stage(
      const MatrixFree<dim, number>                    &matrix_free,
      LinearAlgebra::distributed::Vector<number>       &dst,
      const LinearAlgebra::distributed::Vector<number> &src,
      const std::pair<unsigned int, unsigned int>      &face_range) const;

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
     * The local face applier computing the residual contribution of the inner faces.
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
     * The local cell boundary face computing the residual contribution of the boundary faces.
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
     * When computing the residual this factor defines a scaling factor for the right-hand side.
     * The final residual is then given by R=y'-a*f(y), where 'a' is the factor defined by thi
     * variable.
     */
    mutable number residual_rhs_scaling_factor = 1.;

    /**
     */
    mutable const VectorType *intermediate_explicit_solution = nullptr;

    /**
     * This vector is used in the approximation of the jacobian by finite differences. Here the
     * vector stores the residual with a disturbed input.
     */
    mutable VectorType disturbed_residual;

    /**
     * Current time step size. This needs to be stored as this value is required by the local cell
     * appliers.
     */
    mutable number current_time_increment;

    CompressibleFlowScratchData<dim, number> &flow_scratch_data;

    CompressibleFlowConvectiveKernels<dim, number> convective_terms;

    CompressibleFlowViscousKernels<dim, number> viscous_terms;
  };
} // namespace MeltPoolDG::Flow
