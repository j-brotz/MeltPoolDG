#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

#include <utility>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim, typename number>
  class CompressibleFlowOperatorBase
  {
  protected:
    using VectorType = LinearAlgebra::distributed::Vector<number>;

  public:
    using ConservedVariablesType     = Tensor<1, dim + 2, VectorizedArray<number>>;
    using ConservedVariablesGradType = Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>>;

    CompressibleFlowOperatorBase(const CompressibleFlowData &compressible_flow_data_in,
                                 const ScratchData<dim>     &scratch_data_in,
                                 unsigned int                comp_flow_dof_idx_in  = 0,
                                 unsigned int                comp_flow_quad_idx_in = 0);

    virtual ~CompressibleFlowOperatorBase() = default;

    /**
     * Reinit function which needs to be called before the apply_operator() function. This function
     * computes the interior penalty parameter.
     */
    virtual void
    reinit();

    /**
     * Central function of the operator class which needs to be overridden by all derived classes.
     * This function shall provide the required computations required by a suitable time integrator
     * class.
     */
    virtual void
    apply_operator(number                                                 time,
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func) const = 0;

    /**
     * Set an inflow boundary conditions for all boundary ids occurring in the given std::map and
     * applies the corresponding function at this boundary. This corresponds to a Dirichlet boundary
     * condition to all primary variables.
     *
     * @param inflow_bc Map of boundary ids and corresponding functions for inflow boundaries.
     */
    void
    set_inflow_boundary(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &inflow_bc);

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
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
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
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &outflow_fixed_energy_bc);

    /**
     * Set a slip wall boundary condition for all boundary ids occurring in the given std::map,
     * where the corresponding functions are not used for the boundary condition. This represents a
     * symmetry condition for the momentum (normal velocity results to zero).
     *
     * @param slip_wall_bc Map of boundary ids and corresponding functions for the
     * outflow boundaries.
     */
    void
    set_slip_wall_boundary(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &slip_wall_bc);

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
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &no_slip_wall_bc);

    /**
     * Set a body force, e.g. gravity, specified by the passed function.
     *
     * @param body_force_in Function specifying the body force.
     */
    void
    set_body_force(std::unique_ptr<Function<dim>> body_force_in);

    /**
     * Calculate the velocity from the conserved variables by computing u = (ρu)/ρ.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Current velocity.
     */
    static inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, VectorizedArray<number>>
      calculate_velocity(const ConservedVariablesType &conserved_variables);

    /**
     * Calculate the gradient of the  velocity from the conserved variables and their gradients by
     * computing grad(u) = 1/ρ * (grad(ρu) - u*grad(ρ)).
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the velocity.
     */
    static inline DEAL_II_ALWAYS_INLINE //
      Tensor<2, dim, VectorizedArray<number>>
      calculate_grad_velocity(const ConservedVariablesType     &conserved_variables,
                              const ConservedVariablesGradType &grad_conserved_variables);

    /**
     * Calculate the pressure from the conserved variables and their gradients by computing
     * p = (γ-1)*(ρE - 0.5*ρ*||u||^2).
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param gamma Heat capacity ratio.
     *
     * @return Pressure resulting from the values of the conserved variables.
     */
    static inline DEAL_II_ALWAYS_INLINE //
      VectorizedArray<number>
      calculate_pressure(const ConservedVariablesType &conserved_variables, number gamma);

  protected:
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

    void
    local_apply_inverse_mass_matrix(const MatrixFree<dim, number>                    &matrix_free,
                                    LinearAlgebra::distributed::Vector<number>       &dst,
                                    const LinearAlgebra::distributed::Vector<number> &src,
                                    const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Calculate the gradient of the temperature from the conserved variables and their gradients by
     * computing grad(T) = (γ-1)/R * (grad(E) - grad(u)*u).
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the temperature field.
     */
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, VectorizedArray<number>>
      calculate_grad_T(const ConservedVariablesType     &conserved_variables,
                       const ConservedVariablesGradType &grad_conserved_variables) const;

    /**
     * Contracts the given tensor of the gradient of conserved variables with the given normal
     * vector.
     *
     * @param grad Current gradient of the conserved variables.
     * @param normal Normal vector.
     *
     * @return Result of the contraction, i.e. grad*normal.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType
      contract_tensor_with_normal(const ConservedVariablesGradType              &grad,
                                  const Tensor<1, dim, VectorizedArray<number>> &normal) const;

    /**
     * Computes the average of the passed tensors and contracts the corresponding result with the
     * given normal vector. In other words the function computes 0.5*(grad_m+grad_p)*normal.
     *
     * @param grad_p Current gradient of the conserved variables on the outer face.
     * @param grad_p Current gradient of the conserved variables on the inner face.
     * @param normal Normal vector.
     *
     * @return Result of the contraction.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType
      contract_average_tensor_with_normal(
        const ConservedVariablesGradType              &grad_m,
        const ConservedVariablesGradType              &grad_p,
        const Tensor<1, dim, VectorizedArray<number>> &normal) const;

    /**
     * Calculate the convective flux F_c.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Convective flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesGradType
      calculate_convective_flux(const ConservedVariablesType &conserved_variables) const;

    /**
     * Calculate the convective flux F_c^*.
     *
     * @param u_m Current values of the conserved variables on the inner face.
     * @param u_p Current values of the conserved variables on the outer type.
     * @param normal Outer facing normal vector.
     * @param numerical_flux_type Flux type used for the numerical convective flux.
     *
     * @return Convective numerical flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType
      calculate_convective_numerical_flux(const ConservedVariablesType                  &u_m,
                                          const ConservedVariablesType                  &u_p,
                                          const Tensor<1, dim, VectorizedArray<number>> &normal,
                                          const std::string &numerical_flux_type) const;

    /**
     * Calculate the viscous stress tensor τ given by τ = μ*(grad(u)+grad(u)^T-2/3*(grad*u)*I),
     * where μ is the dynamic viscosity and I representing the identity matrix.
     *
     * @param grad_u Current gradient of the velocity field.
     *
     * @return Viscous stress tensor τ.
     */
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<2, dim, VectorizedArray<number>>
      calculate_viscous_stress_tensor(const Tensor<2, dim, VectorizedArray<number>> &grad_u) const;

    /**
     * Calculate the convective flux F_v.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Viscous flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesGradType
      calculate_viscous_flux(const ConservedVariablesType     &conserved_variables,
                             const ConservedVariablesGradType &grad_conserved_variables) const;

    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType
      calculate_viscous_numerical_flux(const ConservedVariablesType                  &u_m,
                                       const ConservedVariablesType                  &u_p,
                                       const ConservedVariablesGradType              &grad_u_m,
                                       const ConservedVariablesGradType              &grad_u_p,
                                       const Tensor<1, dim, VectorizedArray<number>> &normal,
                                       VectorizedArray<number> penalty_parameter) const;

    inline DEAL_II_ALWAYS_INLINE //
      std::pair<ConservedVariablesGradType, ConservedVariablesGradType>
      calculate_viscous_numerical_flux_gradient(
        const ConservedVariablesType                  &u_m,
        const ConservedVariablesType                  &u_p,
        const Tensor<1, dim, VectorizedArray<number>> &normal) const;

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
    get_adjacent_face_values_at_boundary(const Point<dim, VectorizedArray<number>>     &q_point,
                                         const Tensor<1, dim, VectorizedArray<number>> &normal,
                                         unsigned int                                   boundary_id,
                                         const ConservedVariablesType                  &w_m,
                                         ConservedVariablesType                        &w_p,
                                         const ConservedVariablesGradType              &grad_w_m,
                                         ConservedVariablesGradType &grad_w_p) const;


    /**
     * Compute the boundary conditions at the given time, i.e. evaluate the corresponding boundary
     * conditions at that time.
     *
     * @param time Current time at which the boundary conditions are evaluated.
     */
    void
    update_boundary_conditions(number time) const;

    const CompressibleFlowData &comp_flow_data;

    const ScratchData<dim> &scratch_data;

    const unsigned int comp_flow_dof_idx  = 0;
    const unsigned int comp_flow_quad_idx = 0;

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
  };



  /*****************************************************************************
   * Inlined function definitions
   * **************************************************************************/
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<number>>
    CompressibleFlowOperatorBase<dim, number>::calculate_velocity(
      const ConservedVariablesType &conserved_variables)
  {
    const VectorizedArray<number> inverse_density =
      VectorizedArray<number>(1.) / conserved_variables[0];

    Tensor<1, dim, VectorizedArray<number>> velocity;
    for (unsigned int d = 0; d < dim; ++d)
      velocity[d] = conserved_variables[1 + d] * inverse_density;

    return velocity;
  }


  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, VectorizedArray<number>>
    CompressibleFlowOperatorBase<dim, number>::calculate_grad_velocity(
      const ConservedVariablesType     &conserved_variables,
      const ConservedVariablesGradType &grad_conserved_variables)
  {
    const VectorizedArray<number> inverse_density =
      VectorizedArray<number>(1.) / conserved_variables[0];
    const Tensor<1, dim, VectorizedArray<number>> velocity =
      calculate_velocity(conserved_variables);

    Tensor<1, dim, VectorizedArray<number>> grad_rho;
    for (unsigned int d = 0; d < dim; ++d)
      grad_rho[d] = grad_conserved_variables[0][d];

    Tensor<2, dim, VectorizedArray<number>> grad_rho_velocity;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        grad_rho_velocity[d][e] = grad_conserved_variables[1 + d][e];

    Tensor<2, dim, VectorizedArray<number>> grad_velocity;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        grad_velocity[d][e] =
          inverse_density * (grad_rho_velocity[d][e] - velocity[d] * grad_rho[e]);

    return grad_velocity;
  }


  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<number>
    CompressibleFlowOperatorBase<dim, number>::calculate_pressure(
      const ConservedVariablesType &conserved_variables,
      number                        gamma)
  {
    const Tensor<1, dim, VectorizedArray<number>> velocity =
      calculate_velocity(conserved_variables);
    return (gamma - 1.) * (conserved_variables[dim + 1] -
                           conserved_variables[0] * 0.5 * scalar_product(velocity, velocity));
  }


  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<number>>
    CompressibleFlowOperatorBase<dim, number>::calculate_grad_T(
      const ConservedVariablesType     &conserved_variables,
      const ConservedVariablesGradType &grad_conserved_variables) const
  {
    const Tensor<1, dim, VectorizedArray<number>> u = calculate_velocity(conserved_variables);
    const Tensor<2, dim, VectorizedArray<number>> grad_u =
      calculate_grad_velocity(conserved_variables, grad_conserved_variables);
    const VectorizedArray<number>                 rho      = conserved_variables[0];
    const VectorizedArray<number>                 inv_rho  = VectorizedArray<number>(1.) / rho;
    const Tensor<1, dim, VectorizedArray<number>> grad_rho = grad_conserved_variables[0];
    const Tensor<1, dim, VectorizedArray<number>> grad_E =
      inv_rho *
      (grad_conserved_variables[dim + 1] - inv_rho * conserved_variables[dim + 1] * grad_rho);
    return (comp_flow_data.gamma - 1.0) / comp_flow_data.specific_gas_constant *
           (grad_E - grad_u * u);
  }


  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim + 2, VectorizedArray<number>>
    CompressibleFlowOperatorBase<dim, number>::contract_tensor_with_normal(
      const ConservedVariablesGradType                      &grad,
      const dealii::Tensor<1, dim, VectorizedArray<number>> &normal) const
  {
    ConservedVariablesType result;

    result[0] = grad[0] * normal;

    for (unsigned int e = 0; e < dim; ++e)
      result[e + 1] = grad[e + 1] * normal;

    result[dim + 1] = grad[dim + 1] * normal;

    return result;
  }


  template <int dim, typename number>
  DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim + 2, VectorizedArray<number>>
    CompressibleFlowOperatorBase<dim, number>::contract_average_tensor_with_normal(
      const ConservedVariablesGradType                      &grad_m,
      const ConservedVariablesGradType                      &grad_p,
      const dealii::Tensor<1, dim, VectorizedArray<number>> &normal) const
  {
    ConservedVariablesType result;

    result[0] = (grad_m[0] + grad_p[0]) * normal;

    for (unsigned int e = 0; e < dim; ++e)
      result[e + 1] = (grad_m[e + 1] + grad_p[e + 1]) * normal;

    result[dim + 1] = (grad_m[dim + 1] + grad_p[dim + 1]) * normal;

    return 0.5 * result;
  }


  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>>
    CompressibleFlowOperatorBase<dim, number>::calculate_convective_flux(
      const ConservedVariablesType &conserved_variables) const
  {
    const Tensor<1, dim, VectorizedArray<number>> velocity =
      calculate_velocity(conserved_variables);
    const VectorizedArray<number> pressure =
      calculate_pressure(conserved_variables, comp_flow_data.gamma);

    ConservedVariablesGradType flux;
    for (unsigned int d = 0; d < dim; ++d)
      {
        flux[0][d] = conserved_variables[1 + d];
        for (unsigned int e = 0; e < dim; ++e)
          flux[e + 1][d] = conserved_variables[e + 1] * velocity[d];
        flux[d + 1][d] += pressure;
        flux[dim + 1][d] = velocity[d] * (conserved_variables[dim + 1] + pressure);
      }
    return flux;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim + 2, VectorizedArray<number>>
    CompressibleFlowOperatorBase<dim, number>::calculate_convective_numerical_flux(
      const ConservedVariablesType                  &u_m,
      const ConservedVariablesType                  &u_p,
      const Tensor<1, dim, VectorizedArray<number>> &normal,
      const std::string                             &numerical_flux_type) const
  {
    const auto velocity_m = calculate_velocity(u_m);
    const auto velocity_p = calculate_velocity(u_p);

    const auto pressure_m = calculate_pressure(u_m, comp_flow_data.gamma);
    const auto pressure_p = calculate_pressure(u_p, comp_flow_data.gamma);

    const auto flux_m = calculate_convective_flux(u_m);
    const auto flux_p = calculate_convective_flux(u_p);

    if (numerical_flux_type == "lax_friedrichs_modified")
      {
        const auto lambda =
          0.5 * std::sqrt(std::max(velocity_p.norm_square() +
                                     std::abs(comp_flow_data.gamma * pressure_p * (1. / u_p[0])),
                                   velocity_m.norm_square() +
                                     std::abs(comp_flow_data.gamma * pressure_m * (1. / u_m[0]))));

        return contract_average_tensor_with_normal(flux_m, flux_p, normal) +
               0.5 * lambda * (u_m - u_p);
      }
    else if (numerical_flux_type == "lax_friedrichs_exact")
      {
        const auto lambda =
          std::max(std::abs(velocity_p * normal) +
                     std::sqrt(comp_flow_data.gamma * pressure_p * (1. / u_p[0])),
                   std::abs(velocity_m * normal) +
                     std::sqrt(comp_flow_data.gamma * pressure_m * (1. / u_m[0])));

        return contract_average_tensor_with_normal(flux_m, flux_p, normal) +
               0.5 * lambda * (u_m - u_p);
      }
    else if (numerical_flux_type == "harten_lax_vanleer")
      {
        const auto avg_velocity_normal = 0.5 * ((velocity_m + velocity_p) * normal);
        const auto avg_c               = std::sqrt(std::abs(
          0.5 * comp_flow_data.gamma * (pressure_p * (1. / u_p[0]) + pressure_m * (1. / u_m[0]))));
        const VectorizedArray<number> s_pos =
          std::max(VectorizedArray<number>(), avg_velocity_normal + avg_c);
        const VectorizedArray<number> s_neg =
          std::min(VectorizedArray<number>(), avg_velocity_normal - avg_c);
        const VectorizedArray<number> inverse_s = VectorizedArray<number>(1.) / (s_pos - s_neg);

        return inverse_s * ((s_pos * contract_tensor_with_normal(flux_m, normal) -
                             s_neg * contract_tensor_with_normal(flux_p, normal)) -
                            s_pos * s_neg * (u_m - u_p));
      }
    else
      {
        Assert(false, ExcNotImplemented());
        return {};
      }
  }


  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, VectorizedArray<number>>
    CompressibleFlowOperatorBase<dim, number>::calculate_viscous_stress_tensor(
      const Tensor<2, dim, VectorizedArray<number>> &grad_u) const
  {
    const VectorizedArray<number> div_u = (2. / 3.) * trace(grad_u);

    Tensor<2, dim, VectorizedArray<number>> out;
    for (unsigned int d = 0; d < dim; ++d)
      {
        for (unsigned int e = 0; e < dim; ++e)
          out[d][e] = comp_flow_data.dynamic_viscosity * (grad_u[d][e] + grad_u[e][d]);
        out[d][d] -= comp_flow_data.dynamic_viscosity * div_u;
      }

    return out;
  }



  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>>
    CompressibleFlowOperatorBase<dim, number>::calculate_viscous_flux(
      const ConservedVariablesType     &conserved_variables,
      const ConservedVariablesGradType &grad_conserved_variables) const
  {
    const Tensor<1, dim, VectorizedArray<number>> velocity =
      calculate_velocity(conserved_variables);

    const auto grad_u = calculate_grad_velocity(conserved_variables, grad_conserved_variables);

    const Tensor<2, dim, VectorizedArray<number>> viscous_stress =
      calculate_viscous_stress_tensor(grad_u);

    const Tensor<1, dim, VectorizedArray<number>> neg_heat_flux =
      comp_flow_data.thermal_conductivity *
      calculate_grad_T(conserved_variables, grad_conserved_variables);

    ConservedVariablesGradType flux;
    for (unsigned int d = 0; d < dim; ++d)
      {
        // density
        flux[0][d] = 0.0;

        // momentum
        for (unsigned int e = 0; e < dim; ++e)
          flux[e + 1][d] = viscous_stress[e][d];

        // energy
        flux[dim + 1][d] = neg_heat_flux[d];

        for (unsigned int e = 0; e < dim; ++e)
          flux[dim + 1][d] += velocity[e] * viscous_stress[d][e];
      }

    return flux;
  }



  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim + 2, VectorizedArray<number>>
    CompressibleFlowOperatorBase<dim, number>::calculate_viscous_numerical_flux(
      const ConservedVariablesType                  &u_m,
      const ConservedVariablesType                  &u_p,
      const ConservedVariablesGradType              &grad_u_m,
      const ConservedVariablesGradType              &grad_u_p,
      const Tensor<1, dim, VectorizedArray<number>> &normal,
      const VectorizedArray<number>                  penalty_parameter) const
  {
    const auto flux_m = calculate_viscous_flux(u_m, grad_u_m);

    const auto flux_p = calculate_viscous_flux(u_p, grad_u_p);

    return contract_average_tensor_with_normal(flux_m, flux_p, normal) -
           penalty_parameter * comp_flow_data.dynamic_viscosity / comp_flow_data.reference_density *
             (u_m - u_p);
  }



  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>>,
              Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>>>
    CompressibleFlowOperatorBase<dim, number>::calculate_viscous_numerical_flux_gradient(
      const ConservedVariablesType                  &u_m,
      const ConservedVariablesType                  &u_p,
      const Tensor<1, dim, VectorizedArray<number>> &normal) const
  {
    ConservedVariablesGradType jump_u;
    for (unsigned int e = 0; e < dim + 2; ++e)
      for (unsigned int d = 0; d < dim; ++d)
        jump_u[e][d] = (u_m[e] - u_p[e]) * normal[d];

    // use jumps instead of gradients for evaluating the viscous flux
    const ConservedVariablesGradType flux_m = 0.5 * calculate_viscous_flux(u_m, jump_u);
    const ConservedVariablesGradType flux_p = 0.5 * calculate_viscous_flux(u_p, jump_u);

    return std::make_pair(flux_m, flux_p);
  }
} // namespace MeltPoolDG::Flow
