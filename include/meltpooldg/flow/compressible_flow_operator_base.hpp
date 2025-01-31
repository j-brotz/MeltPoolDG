#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/compressible_flow_calculator_functions.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <concepts>
#include <memory>
#include <tuple>
#include <utility>

namespace MeltPoolDG::Flow
{
  template <typename evaluator_type,
            int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType>
  concept CellEvaluatorType =
    std::is_base_of_v<dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>,
                      evaluator_type> or
    std::is_base_of_v<dealii::FEPointEvaluation<n_components, dim, dim, VectorizedArrayType>,
                      evaluator_type>;

  template <typename evaluator_type,
            int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType>
  concept FaceEvaluatorType =
    std::is_base_of_v<
      dealii::FEFaceEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>,
      evaluator_type> or
    std::is_base_of_v<dealii::FEFacePointEvaluation<n_components, dim, dim, VectorizedArrayType>,
                      evaluator_type>;

  template <int dim, typename number>
  class CompressibleFlowOperatorBase
  {
  protected:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    using ConservedVariablesType = dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>;
    using ConservedVariablesGradType =
      dealii::Tensor<1, dim + 2, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;

    CompressibleFlowOperatorBase(
      const CompressibleFlowData                     &compressible_flow_data_in,
      const ScratchData<dim>                         &scratch_data_in,
      ::TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
      unsigned int                                    comp_flow_dof_idx_in  = 0,
      unsigned int                                    comp_flow_quad_idx_in = 0);

    virtual ~CompressibleFlowOperatorBase() = default;

    /**
     * Reinit function which needs to be called before the apply_operator() function. This function
     * computes the interior penalty parameter.
     */
    virtual void
    reinit();

    /**
     * Core function of the operator class, intended to be overridden by all derived classes.
     *
     * This function executes the necessary computations to perform a single time step. For more
     * details it is referred to the derived class' documentation.
     *
     * @param current_time Current simulation time.
     * @param time_step Current time step size.
     * @param pre_processing Preprocessing function called before the time integration or before
     * each stage during the time integration. The exact implementation depends on the used time
     * inegrator.
     * @param post_processing Postprocessing function called after the time integration or after
     * each stage during the time integration. The exact implementation depends on the used time
     * inegrator.
     */
    virtual void
    advance_time_step(
      number                                                        current_time,
      number                                                        time_step,
      std::function<void(number, VectorType &, const VectorType &)> pre_processing  = {},
      std::function<void(number, VectorType &, const VectorType &)> post_processing = {}){};

    /**
     * Set an inflow boundary conditions for all boundary ids occurring in the given std::map and
     * applies the corresponding function at this boundary. This corresponds to a Dirichlet
     * boundary condition to all primary variables.
     *
     * @param inflow_bc Map of boundary ids and corresponding functions for inflow boundaries.
     */
    void
    set_inflow_boundary(const std::map<dealii::types::boundary_id,
                                       std::shared_ptr<dealii::Function<dim>>> &inflow_bc);

    /**
     * Set a subsonic outflow boundary condition for all boundary ids occurring in the given
     * std::map and applies the corresponding function at this boundary. This corresponds to a
     * Dirichlet boundary condition for the static pressure part of the energy. The dynamic part is
     * added according to the locally present velocity values as it is proposed in Hartmann R. and
     * Houston P., An Optimal Order Interior Penalty Discontinuous Galerkin Discretization of the
     * Compressible Navier–Stokes Equations.
     *
     * @param outflow_fixed_pressure_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     */
    void
    set_subsonic_outflow_with_fixed_static_pressure(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &outflow_fixed_pressure_bc);

    /**
     * Set a subsonic outflow boundary condition with prescribed energy for all boundary ids
     * occurring in the given std::map and applies the corresponding function at this boundary. This
     * corresponds to a Dirichlet boundary condition for the energy.
     *
     * @param outflow_fixed_energy_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     */
    void
    set_subsonic_outflow_with_fixed_energy(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &outflow_fixed_energy_bc);

    /**
     * Set a slip wall boundary condition for all boundary ids occurring in the given std::map,
     * where the corresponding functions are not used for the boundary condition. This represents a
     * symmetry condition for the momentum (normal velocity results to zero).
     *
     * @param slip_wall_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     */
    void
    set_slip_wall_boundary(const std::map<dealii::types::boundary_id,
                                          std::shared_ptr<dealii::Function<dim>>> &slip_wall_bc);

    /**
     * Set an adiabatic no-slip wall boundary condition for all boundary ids occurring in the given
     * std::map, where the corresponding functions are not used for the boundary condition. This
     * represents a homogeneous Dirichlet boundary condition for the momentum and a homogeneous
     * Neumann boundary condition for the energy.
     *
     * @param no_slip_wall_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     */
    void
    set_no_slip_adiabatic_wall_boundary(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &no_slip_wall_bc);

    /**
     * Set a body force, e.g. gravity, specified by the passed function.
     *
     * @param body_force_in Function specifying the body force.
     */
    void
    set_body_force(std::unique_ptr<dealii::Function<dim>> body_force_in);

    /**
     * Set the inflow field function in the case of an unfitted inflow boundary.
     */
    // TODO: eliminate this function from operator base class?
    virtual void
    set_inflow_field_unfitted_boundary(std::shared_ptr<Function<dim>> & /*inflow_function*/){

    };

    /**
     * Set the velocity function in the case of an unfitted (rigid) moving object.
     */
    // TODO: eliminate this function from operator base class?
    virtual void
    set_unfitted_object_velocity(std::shared_ptr<Function<dim>> & /*velocity_function*/){

    };

    /**
     * Compute the boundary conditions at the given time, i.e. evaluate the corresponding boundary
     * conditions at that time.
     *
     * @param time Current time at which the boundary conditions are evaluated.
     */
    void
    update_boundary_conditions(number time) const;

    /**
     * Function for the matrix-free right-hand side vector evaluation.
     *
     * @param time Current simulation time.
     * @param time_step Current time step size.
     * @param dst Vector where the computed right-hand side rhs(src) is stored.
     * @param src The solution vector at the current time.
     */
    virtual void
    create_rhs(const number     &time,
               const number     &time_step,
               VectorType       &dst,
               const VectorType &src) const {};

    /**
     * Function for the matrix-free matrix-vector product evaluation.
     *
     * @param dst Vector where the computed evaluated vector-matrix product is stored.
     * @param src The solution vector at the current time.
     */
    virtual void
    vmult(VectorType &dst, const VectorType &src) const {};

    /**
     * Function which computes the inverse time step size.
     */
    void
    compute_inverse_time_step(const number &time_step)
    {
      AssertThrow(time_step > 0., ExcMessage("Time step size must be larger than 0!"));
      inv_time_step = 1. / time_step;
    };

  protected:
    /**
     * Apply the inverse of the mass matrix to the given dof vector.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector where the solution is stored.
     * @param src Current solution of the primary variables.
     * @param cell_range Cell range on which the inverse mass matrix is applied.
     */
    void
    local_apply_inverse_mass_matrix(const MatrixFree<dim, number>                    &matrix_free,
                                    LinearAlgebra::distributed::Vector<number>       &dst,
                                    const LinearAlgebra::distributed::Vector<number> &src,
                                    const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Kernel of the local cell applier for the right-hand side function. This function computes the
     * cell integral contribution to the right hand side for the quadrature point index and the
     * corresponding FE evaluator.
     *
     * @param evaluator FE-evaluator object reinitialized on the current cell batch.
     * @param q Index of the quadrature point.
     * @param constant_body_force Value of the body force. If the body force is not constant the
     * pointer must be set to nullptr.
     *
     * @return Tuple, containing the flux, weighted with the value of the test function, as first
     * argument, and the flux, weighted with the gradient of the test function, as second argument.
     */
    template <CellEvaluatorType<dim,
                                dim + 2,
                                number,
                                dealii::VectorizedArray<number>> Integrator>
    inline DEAL_II_ALWAYS_INLINE //
      std::tuple<ConservedVariablesType, ConservedVariablesGradType>
      rhs_cell_integral_kernel(
        const Integrator                                              &evaluator,
        unsigned int                                                   q,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> *constant_body_force) const;

    /**
     * Kernel of the local inner face applier for the right-hand side function. This function
     * computes the face integral contribution of inner faces to the right hand side for the
     * quadrature point index and the corresponding FE evaluator.
     *
     * @param evaluator_m FE-evaluator object reinitialized on the current (inside) face batch.
     * @param evaluator_p FE-evaluator object reinitialized on the current (outside) face batch.
     * @param q Index of the quadrature point.
     * @param penalty_parameter Value of the symmetric interior penalty parameter on the face.
     *
     * @return Tuple, which containing the fluxes for the inside and outside faces, weighted with
     * the value of the test functions, as first two arguments, and the fluxes for the inside and
     * outside faces, weighted with the gradient of the test functions, as the third and fourth
     * argument.
     */
    template <FaceEvaluatorType<dim,
                                dim + 2,
                                number,
                                dealii::VectorizedArray<number>> Integrator>
    inline DEAL_II_ALWAYS_INLINE //
      std::tuple<ConservedVariablesType,
                 ConservedVariablesType,
                 ConservedVariablesGradType,
                 ConservedVariablesGradType>
      rhs_face_integral_kernel(const Integrator               &evaluator_m,
                               const Integrator               &evaluator_p,
                               unsigned                        q,
                               dealii::VectorizedArray<number> penalty_parameter) const;

    /**
     * Kernel of the local boundary face applier for the right-hand side function. This function
     * computes the face integral contribution of boundary faces to the right hand side for the
     * quadrature point index and the corresponding FE evaluator.
     *
     * @param evaluator_m FE-evaluator object reinitialized on the current (inner) face batch.
     * @param q Index of the quadrature point.
     * @param boundary_id ID of the boundary.
     * @param penalty_parameter Value of the symmetric interior penalty parameter on the face.
     *
     * @return Tuple, containing the flux for the boundary face, weighted with the value of the test
     * function, as first argument, and the flux for the boundary face, weighted with the gradient
     * of the test function, as second argument.
     */
    template <FaceEvaluatorType<dim,
                                dim + 2,
                                number,
                                dealii::VectorizedArray<number>> Integrator>
    inline DEAL_II_ALWAYS_INLINE //
      std::tuple<ConservedVariablesType, ConservedVariablesGradType>
      rhs_boundary_face_integral_kernel(const Integrator               &evaluator_m,
                                        unsigned int                    q,
                                        const unsigned int              face,
                                        dealii::VectorizedArray<number> penalty_parameter) const;

    /**
     * This function sets the corresponding values on the fictional outer face if the face is
     * located at a boundary.
     *
     * @param q_point Location of the quadrature points at which the values shall be computed.
     * @param normal Outer facing normal vector.
     * @param boundary_id ID of the boundary.
     * @param w_m Conserved variables on the inner face.
     * @param w_p Location where the corresponding boundary values are stored.
     * @param grad_w_m Gradient of the conserved variables on the inner face.
     * @param grad_w_p Location where the corresponding gradients of the boundary values shall
     * be stored.
     */
    void
    get_adjacent_face_values_at_boundary(
      const Point<dim, dealii::VectorizedArray<number>>             &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      unsigned int                                                   boundary_id,
      const ConservedVariablesType                                  &w_m,
      ConservedVariablesType                                        &w_p,
      const ConservedVariablesGradType                              &grad_w_m,
      ConservedVariablesGradType                                    &grad_w_p) const;

    /**
     * This function makes the necessary function call to compute the interior penalty parameter in
     * the case that the viscosity is greater than zero.
     */
    void
    calculate_interior_penalty_parameter();

    const CompressibleFlowData &comp_flow_data;

    const ScratchData<dim> &scratch_data;

    ::TimeIntegration::SolutionHistory<VectorType> &solution_history;

    const unsigned int comp_flow_dof_idx  = 0;
    const unsigned int comp_flow_quad_idx = 0;

    const CompressibleFlowCalculators<dim, number> calculator_functions;

    AlignedVector<VectorizedArray<number>> interior_penalty_parameter;

    //! Boundary conditions
    mutable std::map<types::boundary_id, std::shared_ptr<Function<dim>>> inflow_boundaries;
    mutable std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
      subsonic_outflow_fixed_pressure;
    mutable std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
                                 subsonic_outflow_fixed_energy;
    std::set<types::boundary_id> slip_wall_boundaries;
    std::set<types::boundary_id> no_slip_adiabatic_wall_boundaries;

    mutable std::unique_ptr<Function<dim>> body_force;

    // inverse time step
    number inv_time_step = dealii::numbers::invalid_double;
  };



  /*****************************************************************************
   * Inlined function definitions
   * **************************************************************************/
  template <int dim, typename number>
  template <CellEvaluatorType<dim,
                              dim + 2,
                              number,
                              dealii::VectorizedArray<number>> Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesType,
               typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesGradType>
    CompressibleFlowOperatorBase<dim, number>::rhs_cell_integral_kernel(
      const Integrator                                              &evaluator,
      const unsigned int                                             q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> *constant_body_force) const
  {
    const auto w_q = evaluator.get_value(q);

    auto flux = calculator_functions.calculate_convective_flux(w_q);

    if (this->comp_flow_data.dynamic_viscosity > 0)
      {
        const auto grad_w_q = evaluator.get_gradient(q);
        flux -= calculator_functions.calculate_viscous_flux(w_q, grad_w_q);
      }

    dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>> forcing;

    if (this->body_force.get() != nullptr)
      {
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> force =
          constant_body_force ?
            *constant_body_force :
            VectorTools::evaluate_function_at_vectorized_points(*this->body_force,
                                                                evaluator.quadrature_point(q));
        for (unsigned int d = 0; d < dim; ++d)
          forcing[d + 1] = w_q[0] * force[d];
        for (unsigned int d = 0; d < dim; ++d)
          forcing[dim + 1] += force[d] * w_q[d + 1];
      }

    return {forcing, flux};
  }


  template <int dim, typename number>
  template <FaceEvaluatorType<dim,
                              dim + 2,
                              number,
                              dealii::VectorizedArray<number>> Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesType,
               typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesType,
               typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesGradType,
               typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesGradType>
    CompressibleFlowOperatorBase<dim, number>::rhs_face_integral_kernel(
      const Integrator                     &evaluator_m,
      const Integrator                     &evaluator_p,
      const unsigned int                    q,
      const dealii::VectorizedArray<number> penalty_parameter) const
  {
    auto numerical_flux =
      calculator_functions.calculate_convective_numerical_flux(evaluator_m.get_value(q),
                                                               evaluator_p.get_value(q),
                                                               evaluator_m.normal_vector(q));

    if (this->comp_flow_data.dynamic_viscosity > 0)
      numerical_flux -=
        calculator_functions.calculate_viscous_numerical_flux(evaluator_m.get_value(q),
                                                              evaluator_p.get_value(q),
                                                              evaluator_m.get_gradient(q),
                                                              evaluator_p.get_gradient(q),
                                                              evaluator_m.normal_vector(q),
                                                              penalty_parameter);

    std::pair<ConservedVariablesGradType, ConservedVariablesGradType> viscous_numerical_flux;

    // interior penalty
    if (this->comp_flow_data.dynamic_viscosity > 0)
      {
        viscous_numerical_flux = calculator_functions.calculate_viscous_numerical_flux_gradient(
          evaluator_m.get_value(q), evaluator_p.get_value(q), evaluator_m.normal_vector(q));
      }

    return {-numerical_flux,
            numerical_flux,
            viscous_numerical_flux.first,
            viscous_numerical_flux.second};
  }


  template <int dim, typename number>
  template <FaceEvaluatorType<dim,
                              dim + 2,
                              number,
                              dealii::VectorizedArray<number>> Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesType,
               typename CompressibleFlowOperatorBase<dim, number>::ConservedVariablesGradType>
    CompressibleFlowOperatorBase<dim, number>::rhs_boundary_face_integral_kernel(
      const Integrator                     &evaluator_m,
      const unsigned int                    q,
      const unsigned int                    face,
      const dealii::VectorizedArray<number> penalty_parameter) const
  {
    const auto w_m      = evaluator_m.get_value(q);
    const auto normal   = evaluator_m.normal_vector(q);
    const auto grad_w_m = evaluator_m.get_gradient(q);

    ConservedVariablesType     w_p;
    ConservedVariablesGradType grad_w_p;

    const auto boundary_id = scratch_data.get_matrix_free().get_boundary_id(face);

    this->get_adjacent_face_values_at_boundary(
      evaluator_m.quadrature_point(q), normal, boundary_id, w_m, w_p, grad_w_m, grad_w_p);

    auto flux = calculator_functions.calculate_convective_numerical_flux(w_m, w_p, normal);

    if (this->comp_flow_data.dynamic_viscosity > 0)
      flux -= calculator_functions.calculate_viscous_numerical_flux(
        w_m, w_p, grad_w_m, grad_w_p, normal, penalty_parameter);

    ConservedVariablesGradType numerical_flux_gradient;

    if (this->comp_flow_data.dynamic_viscosity > 0)
      {
        numerical_flux_gradient =
          calculator_functions.calculate_viscous_numerical_flux_gradient(w_m, w_p, normal).first;
      }

    return {-flux, numerical_flux_gradient};
  }
} // namespace MeltPoolDG::Flow
