#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/material_data.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <map>
#include <memory>
#include <set>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * An enum for the various boundary conditions supported by the compressible flow solver.
   */
  BETTER_ENUM(BoundaryConditionType,
              char,
              inflow,
              slip_wall,
              no_slip_wall,
              subsonic_outflow_fixed_energy,
              subsonic_outflow_fixed_pressure);

  /**
   * @brief Helper class taking care of all boundary condition related computations for the
   * compressible flow solver.
   *
   * This class provides an interface to set, manage, and evaluate a variety of physical boundary
   * conditions that may arise in simulations of compressible Navier-Stokes flows.
   *
   * Supported boundary condition types:
   * - Inflow
   * - Slip wall
   * - No-slip adiabatic wall
   * - Subsonic outflow with fixed pressure
   * - Subsonic outflow with fixed energy
   */
  template <int dim, typename number>
  class BoundaryConditions
  {
  public:
    using ConservedVariablesType = dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>;
    using ConservedVariablesGradType =
      dealii::Tensor<1, dim + 2, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;
    using BoundaryType = BoundaryConditionType;

    using VectorizedArrayType = dealii::VectorizedArray<number>;

    /**
     * @brief Update the boundary conditions.
     *
     * This means if the boundary conditions are time-dependent functions compute and store their
     * values for the given time.
     *
     * @param time Time at which the boundary conditions are evaluated.
     */
    void
    update_boundary_conditions(number time);

    /**
     * @brief Set the compressible flow boundary conditions.
     *
     * Set the boundary conditions by internally calling the function set_boundary_condition for the
     * currently implemented boundary types.
     *
     * @param simulation_case Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name)
    {
      set_boundary_condition(BoundaryConditionType::inflow,
                             simulation_case->get_boundary_condition("inflow", operation_name));

      set_boundary_condition(BoundaryConditionType::subsonic_outflow_fixed_pressure,
                             simulation_case->get_boundary_condition("outflow_fixed_pressure",
                                                                     operation_name));

      set_boundary_condition(BoundaryConditionType::subsonic_outflow_fixed_energy,
                             simulation_case->get_boundary_condition("outflow_fixed_energy",
                                                                     operation_name));

      set_boundary_condition(BoundaryConditionType::slip_wall,
                             simulation_case->get_boundary_condition("slip_wall", operation_name));

      set_boundary_condition(BoundaryConditionType::no_slip_wall,
                             simulation_case->get_boundary_condition("no_slip_wall",
                                                                     operation_name));
    }

    /**
     * @brief Set a specific boundary condition and store it internally.
     *
     * @param boundary_condition Type of the boundary condition.
     * @param boundary_condition_function Map containing the boundary id at which the condition
     * applies and an (optional) function describing the possibly time and space dependent boundary
     * condition.
     */
    void
    set_boundary_condition(
      BoundaryConditionType boundary_condition,
      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        boundary_condition_function = {});

    /**
     * @brief Get the type of the boundary condition at a specific domain boundary.
     *
     * @param boundary_id Boundary id of the boundary of interest.
     *
     * @return Type of the boundary condition at the boundary with the passed boundary id.
     *
     * @throws Exception if the boundary with the corresponding boundary id has no
     * boundary condition set.
     */
    BoundaryConditionType
    get_boundary_type(const dealii::types::boundary_id boundary_id) const
    {
      if (inflow_boundaries.contains(boundary_id))
        return BoundaryConditionType::inflow;
      if (slip_wall_boundaries.contains(boundary_id))
        return BoundaryConditionType::slip_wall;
      if (no_slip_adiabatic_wall_boundaries.contains(boundary_id))
        return BoundaryConditionType::no_slip_wall;
      if (subsonic_outflow_fixed_energy.contains(boundary_id))
        return BoundaryConditionType::subsonic_outflow_fixed_energy;
      if (subsonic_outflow_fixed_pressure.contains(boundary_id))
        return BoundaryConditionType::subsonic_outflow_fixed_pressure;
      AssertThrow(false,
                  dealii::ExcMessage(
                    "There is no compressible flow boundary set at the boundary with boundary id " +
                    std::to_string(boundary_id) + "!"));
    }

    /**
     * @brief Compute the prescribed boundary value at a specific location on the boundary.
     *
     * @param boundary_id Boundary id of the boundary of interest.
     * @param boundary_condition Type of the boundary condition.
     * @param location Coordinates at which the boundary value is computed.
     * @param component If the boundary value is a vector, the component of the vector which
     * is returned.
     *
     * @return Prescribed boundary value.
     */
    dealii::VectorizedArray<number>
    get_boundary_value(dealii::types::boundary_id boundary_id,
                       BoundaryConditionType      boundary_condition,
                       const dealii::Point<dim, dealii::VectorizedArray<number>> &location,
                       unsigned                                                   component) const;

    /**
     * @brief Same as above but returns the complete vector.
     *
     * @param boundary_id Boundary id of the boundary of interest.
     * @param boundary_condition Type of the boundary condition.
     * @param location Coordinates at which the boundary value is computed.
     *
     * @return Prescribed boundary value.
     */
    dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>
    get_boundary_value(const dealii::types::boundary_id boundary_id,
                       const BoundaryConditionType      boundary_condition,
                       const dealii::Point<dim, dealii::VectorizedArray<number>> &location) const;

    /**
     * @brief This function sets the corresponding values on the fictional outer face if the face is
     * located at a boundary.
     *
     * @param q_point Location of the quadrature points at which the values shall be computed.
     * @param normal Outer facing normal vector.
     * @param boundary_id ID of the boundary.
     * @param w_m Conserved variables on the inner face.
     * @param grad_w_m Gradient of the conserved variables on the inner face.
     * @param material Material class, which contains the material parameters and helper functions
     * for thermodynamic relations.
     * @param is_gas_phase Boolean variable to indicate if the gas phase (default for single-phase
     * case) or the liquid phase is considered.
     *
     * @return Tuple containing the corresponding values on the outer face. The first value being
     * the primary variables, the second the gradient of the primary variables.
     */
    std::tuple<ConservedVariablesType, ConservedVariablesGradType>
    get_boundary_face_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::types::boundary_id                                     boundary_id,
      const ConservedVariablesType                                  &w_m,
      const ConservedVariablesGradType                              &grad_w_m,
      const Material<dim, number>                                   &material,
      bool                                                           is_gas_phase = true) const;

    /**
     * As above but based on views.
     *
     * @param q_point Location of the quadrature points at which the values shall be computed.
     * @param normal Outer facing normal vector.
     * @param boundary_id ID of the boundary.
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     */
    template <typename DofReadView, typename DofWriteView>
    void
    set_conserved_variables_boundary_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::types::boundary_id                                     boundary_id,
      const DofReadView                                             &w_m,
      const DofWriteView                                            &w_p) const;
      
    /**
     * @brief Compute boundary values and gradients, as well as their linearizations for the Jacobian.
     *
     * This function sets the corresponding values on the fictional outer face if the face is
     * located at a boundary. It returns the values and the gradient of the boundary values as well
     * as their linearized version required when for example computing the Jacobian in an implicit
     * scheme.
     *
     * @param q_point Coordinates of the quadrature point.
     * @param normal Outer facing normal vector.
     * @param boundary_id ID of the current boundary.
     * @param w_m Primary variables on the inner face.
     * @param delta_w_m Change in the primary variables on the inner face.
     * @param grad_w_m Gradient of the primary variables on the inner face.
     * @param grad_delta_w_m Gradient of the change in the primary variables on the inner face.
     * @param gamma Heat capacity ratio of the flow field.
     *
     * @return Tuple containing the corresponding values on the outer face. The first value being
     * the primary variables, the second the gradient of the primary variables, the third the change
     * in the primary variables and the fourth the change in the primary variables.
     */
    std::tuple<ConservedVariablesType,
               ConservedVariablesGradType,
               ConservedVariablesType,
               ConservedVariablesGradType>
    get_jacobian_boundary_face_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::types::boundary_id                                     boundary_id,
      const ConservedVariablesType                                  &w_m,
      const ConservedVariablesType                                  &delta_w_m,
      const ConservedVariablesGradType                              &grad_w_m,
      const ConservedVariablesGradType                              &grad_delta_w_m,
      number                                                         gamma) const;

  private:
    /// Maps boundary IDs to the specific boundary functions.
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> inflow_boundaries;
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      subsonic_outflow_fixed_pressure;
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      subsonic_outflow_fixed_energy;

    /// Sets of boundary IDs.
    std::set<dealii::types::boundary_id> slip_wall_boundaries;
    std::set<dealii::types::boundary_id> no_slip_adiabatic_wall_boundaries;
  };
} // namespace MeltPoolDG::CompressibleFlow

template <int dim, typename number>
template <typename DofReadView, typename DofWriteView>
void
MeltPoolDG::CompressibleFlow::BoundaryConditions<dim, number>::
  set_conserved_variables_boundary_value_and_gradient(
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    dealii::types::boundary_id                                     boundary_id,
    const DofReadView                                             &w_m,
    const DofWriteView                                            &w_p) const
{
  if (const BoundaryType boundary_type = get_boundary_type(boundary_id);
      boundary_type == BoundaryType::slip_wall)
    {
      // homogeneous Neumann
      w_p.density()      = w_m.density();
      w_p.grad_density() = -(w_m.grad_density());

      // homogeneous Neumann
      VectorizedArrayType rho_u_dot_n = 0;
      for (unsigned int d = 0; d < dim; ++d)
        rho_u_dot_n += w_m.momentum(d) * normal[d];
      // symmetry
      for (unsigned int d = 0; d < dim; ++d)
        {
          w_p.momentum(d) = w_m.momentum(d) - 2. * rho_u_dot_n * normal[d];
          w_p.grad_momentum(d) =
            w_m.grad_momentum(d) - 2. * scalar_product(w_m.grad_momentum(d), normal) * normal;
        }
      // homogeneous Neumann
      w_p.grad_total_energy() = -(w_m.grad_total_energy());
      w_p.total_energy()      = w_m.total_energy();
    }
  else if (boundary_type == BoundaryType::no_slip_wall)
    {
      // homogeneous Neumann
      w_p.density()      = w_m.density();
      w_p.grad_density() = -(w_m.grad_density());

      // Dirichlet
      for (unsigned int d = 0; d < dim; ++d)
        {
          w_p.momentum(d)      = 0.;
          w_p.grad_momentum(d) = w_m.grad_momentum(d);
        }
      // homogeneous Neumann
      w_p.grad_total_energy() = -(w_m.grad_total_energy());
      w_p.total_energy()      = w_m.total_energy();
    }
  else if (boundary_type == BoundaryType::inflow)
    {
      w_p.density()      = get_boundary_value(boundary_id, BoundaryType::inflow, q_point, 0);
      w_p.grad_density() = w_m.grad_density();

      for (unsigned int d = 0; d < dim; ++d)
        {
          w_p.momentum(d) = get_boundary_value(boundary_id, BoundaryType::inflow, q_point, d + 1);
          w_p.grad_momentum(d) = w_m.grad_momentum(d);
        }

      w_p.total_energy() = get_boundary_value(boundary_id, BoundaryType::inflow, q_point, dim + 1);
      w_p.grad_total_energy() = w_m.grad_total_energy();
    }
  else if (boundary_type == BoundaryType::subsonic_outflow_fixed_pressure)
    {
      // homogeneous Neumann
      w_p.density()      = w_m.density();
      w_p.grad_density() = -(w_m.grad_density());

      // Dirichlet
      auto p_dyn = dealii::VectorizedArray<number>(0.);
      for (unsigned int i = 0; i < dim; ++i)
        p_dyn += w_m.momentum(i) * w_m.momentum(i);

      p_dyn /= (w_m.density() * 2.);
      const dealii::VectorizedArray<number> pressure =
        get_boundary_value(boundary_id, BoundaryType::subsonic_outflow_fixed_pressure, q_point, 0);

      // consider equation of state for computation of inner energy from given pressure
      const dealii::VectorizedArray<number> inner_energy = w_m.inner_energy_from_pressure(pressure);

      w_p.total_energy()      = inner_energy + p_dyn;
      w_p.grad_total_energy() = w_m.grad_total_energy();
    }
  else if (boundary_type == BoundaryType::subsonic_outflow_fixed_energy)
    {
      // homogeneous Neumann
      w_p.density()      = w_m.density();
      w_p.grad_density() = -(w_m.grad_density());
      for (unsigned int i = 0; i < dim; ++i)
        {
          w_p.momentum(i)      = w_m.momentum(i);
          w_p.grad_momentum(i) = -(w_m.grad_momentum(i));
        }

      // Dirichlet
      w_p.total_energy() =
        get_boundary_value(boundary_id, BoundaryType::subsonic_outflow_fixed_energy, q_point, 0);
      w_p.grad_total_energy() = w_m.grad_total_energy();
    }
  else
    {
      std::cout << "ID: " << boundary_id << std::endl;
      std::cout << "Condition:" << boundary_type << std::endl;
      AssertThrow(false,
                  dealii::ExcMessage("Unknown boundary id, did "
                                     "you set a boundary condition for "
                                     "this part of the domain boundary?"));
    }
}
