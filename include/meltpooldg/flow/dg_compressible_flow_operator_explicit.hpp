#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <functional>
#include <memory>
#include <utility>

namespace MeltPoolDG::Flow
{
  template <unsigned int dim, typename number = double, bool is_viscous = true>
  class DGCompressibleFlowOperatorExplicit final : public DGCompressibleFlowOperatorBase<number>
  {
    using VectorType             = dealii::LinearAlgebra::distributed::Vector<number>;
    using ConservedVariablesType = CompressibleFlowTypes::ConservedVariablesType<dim, number>;
    using ConservedVariablesGradType =
      CompressibleFlowTypes::ConservedVariablesGradType<dim, number>;

  public:
    /**
     * Constructor.
     *
     * @param flow_scratch_data Reference to the flow scratch data object (usually owned by the
     * corresponding operation class).
     */
    explicit DGCompressibleFlowOperatorExplicit(
      CompressibleFlowScratchData<dim, number> &flow_scratch_data);

    /**
     * Reinitilaize the internal data structures, i.e. allocate memory for vectors storing temporary
     * solutions.
     */
    void
    reinit() override;

    /**
     * Creates and returns an explicit time integrator object which is set up with the current
     * operator.
     *
     * @param time_integrator_data Reference to the time integrator data object.
     *
     * @return Unique pointer to a time integrator which is templated on the own operator type.
     *
     * @throws If the time integrator type in the time integrator data is not an explicit time
     * integrator.
     */
    std::unique_ptr<TimeIntegratorBase<number>>
    make_problem_specific_time_integrator(const TimeIntegratorData &time_integrator_data) override;

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
     * Local cell applier computing the cell contribution to the rhs if the compressible
     * Navier-Stokes equations are written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param cell_range Cell range which is considered in the applier.
     */
    void
    local_apply_cell(const dealii::MatrixFree<dim, number>                    &matrix_free,
                     dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                     const dealii::LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned int, unsigned int>              &cell_range) const;

    /**
     * Local face applier computing the face contribution to the rhs if the compressible
     * Navier-Stokes equations are written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_apply_face(const dealii::MatrixFree<dim, number>                    &matrix_free,
                     dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                     const dealii::LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned int, unsigned int>              &face_range) const;

    /**
     * Local boundaryface applier computing the boudnary face contribution to the rhs if the
     * compressible Navier-Stokes equations are written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Boundary face range which is considered in the applier.
     */
    void
    local_apply_boundary_face(const dealii::MatrixFree<dim, number>                    &matrix_free,
                              dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                              const dealii::LinearAlgebra::distributed::Vector<number> &src,
                              const std::pair<unsigned int, unsigned int> &face_range) const;

    CompressibleFlowScratchData<dim, number> &flow_scratch_data;

    CompressibleFlowConvectiveKernels<dim, number> convective_terms;

    CompressibleFlowViscousKernels<dim, number> viscous_terms;
  };
} // namespace MeltPoolDG::Flow