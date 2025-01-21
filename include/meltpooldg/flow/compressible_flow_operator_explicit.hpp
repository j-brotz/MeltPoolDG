#pragma once

#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_operator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <unsigned int dim, typename number = double>
  class CompressibleFlowOperatorExplicit final : public CompressibleFlowOperatorBase<dim, number>
  {
    using VectorType = LinearAlgebra::distributed::Vector<number>;

  public:
    /**
     * Constructor.
     *
     * @param comp_flow_data_in Reference to the compressible flow data struct used.
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param solution_history_in Reference to the used solution_history object.
     * @param comp_flow_dof_idx_in Index of the used dof handler in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     */
    CompressibleFlowOperatorExplicit(
      const CompressibleFlowData                     &comp_flow_data_in,
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
     * Computes the value of the function f(y) for the compressible Navier-Stokes equations of the
     * form y' = f(y). From a discretization perspective, f(y) is given by f(y) = M^(-1) * F(y),
     * where M is the mass matrix and F(y) is the sum of all flux contributions: F_v + F_c + F_rhs.
     *
     * @param time The current time at which the function is evaluated.
     * @param dst Vector where the computed value of f(y) is stored.
     * @param src The solution vector, y, at the current time.
     * @param func A function to be executed after f(y) has been computed. This function is applied
     * to the resulting vector in @p dst.
     */
    void
    apply_operator(number                                                 time,
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func) const;

  private:
    /**
     * Local appliers
     */
    void
    local_apply_cell(const MatrixFree<dim, number>                    &matrix_free,
                     LinearAlgebra::distributed::Vector<number>       &dst,
                     const LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void
    local_apply_face(const MatrixFree<dim, number>                    &matrix_free,
                     LinearAlgebra::distributed::Vector<number>       &dst,
                     const LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned int, unsigned int>      &face_range) const;

    void
    local_apply_boundary_face(const MatrixFree<dim, number>                    &matrix_free,
                              LinearAlgebra::distributed::Vector<number>       &dst,
                              const LinearAlgebra::distributed::Vector<number> &src,
                              const std::pair<unsigned int, unsigned int>      &face_range) const;


    /**
     * Explicit time integrator.
     */
    std::unique_ptr<TimeIntegratorBase<number, CompressibleFlowOperatorExplicit<dim, number>>>
      time_integrator;
  };
} // namespace MeltPoolDG::Flow