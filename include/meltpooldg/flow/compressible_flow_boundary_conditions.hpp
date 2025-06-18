#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_material.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <map>
#include <memory>
#include <set>

namespace MeltPoolDG::Flow
{
  /**
   * An enum for the various boundary conditions supported by the compressible flow solver.
   */
  BETTER_ENUM(CompressibleBoundaryConditionType,
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
  class CompressibleFlowBoundaryConditions
  {
  public:
    using ConservedVariablesType = dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>;
    using ConservedVariablesGradType =
      dealii::Tensor<1, dim + 2, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;
    using BoundaryType = CompressibleBoundaryConditionType;

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
     * @param simulation_case dealii::Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name)
    {
      set_boundary_condition(MeltPoolDG::Flow::CompressibleBoundaryConditionType::inflow,
                             simulation_case->get_boundary_condition("inflow", operation_name));

      set_boundary_condition(
        MeltPoolDG::Flow::CompressibleBoundaryConditionType::subsonic_outflow_fixed_pressure,
        simulation_case->get_boundary_condition("outflow_fixed_pressure", operation_name));

      set_boundary_condition(
        MeltPoolDG::Flow::CompressibleBoundaryConditionType::subsonic_outflow_fixed_energy,
        simulation_case->get_boundary_condition("outflow_fixed_energy", operation_name));

      set_boundary_condition(MeltPoolDG::Flow::CompressibleBoundaryConditionType::slip_wall,
                             simulation_case->get_boundary_condition("slip_wall", operation_name));

      set_boundary_condition(MeltPoolDG::Flow::CompressibleBoundaryConditionType::no_slip_wall,
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
      CompressibleBoundaryConditionType boundary_condition,
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
    CompressibleBoundaryConditionType
    get_boundary_type(const dealii::types::boundary_id boundary_id) const
    {
      if (inflow_boundaries.contains(boundary_id))
        return CompressibleBoundaryConditionType::inflow;
      if (slip_wall_boundaries.contains(boundary_id))
        return CompressibleBoundaryConditionType::slip_wall;
      if (no_slip_adiabatic_wall_boundaries.contains(boundary_id))
        return CompressibleBoundaryConditionType::no_slip_wall;
      if (subsonic_outflow_fixed_energy.contains(boundary_id))
        return CompressibleBoundaryConditionType::subsonic_outflow_fixed_energy;
      if (subsonic_outflow_fixed_pressure.contains(boundary_id))
        return CompressibleBoundaryConditionType::subsonic_outflow_fixed_pressure;
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
    get_boundary_value(dealii::types::boundary_id        boundary_id,
                       CompressibleBoundaryConditionType boundary_condition,
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
    get_boundary_value(const dealii::types::boundary_id        boundary_id,
                       const CompressibleBoundaryConditionType boundary_condition,
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
      const CompressibleFlowMaterial<dim, number>                   &material,
      bool                                                           is_gas_phase = true) const;

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
} // namespace MeltPoolDG::Flow
