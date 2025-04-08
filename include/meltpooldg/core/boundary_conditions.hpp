#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

#include <meltpooldg/core/exceptions.hpp>

#include <map>
#include <memory>
#include <ranges>
#include <string>

namespace MeltPoolDG
{
  template <int dim, typename number>
  class BoundaryConditionManager
  {
  private:
    /**
     * @brief Nested map for storing boundary conditions.
     *
     * The map structure is organized as:
     * - `bc[<type>][<id>] = <function>`
     *   - `<type>`: A string representing the type of the boundary condition
     *     (e.g., "dirichlet", "Neumann").
     *   - `<id>`: A unique boundary identifier of type `dealii::types::boundary_id`.
     *   - `<function>`: A shared pointer to a `Function<dim>` object that defines the
     *     behavior of the boundary condition.
     */
    std::map<std::string,
             std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>>
      bc;

  public:
    /**
     * @brief Attach a boundary condition to the manager.
     *
     * Adds a boundary condition specified by its type, boundary ID, and function. Ensures
     * that no duplicate boundary conditions are assigned to the same ID for the same type.
     *
     * @param id_and_function A pair consisting of:
     *        - `dealii::types::boundary_id`: The unique ID for the boundary.
     *        - `std::shared_ptr<dealii::Function<dim>>`: A shared pointer to the function defining
     *          the boundary condition.
     * @param type The type of the boundary condition (e.g., "dirichlet", "neumann").
     *
     * @throws ExcBCAlreadyAssigned If a boundary condition is already assigned to the given ID and
     * type.
     */
    void
    attach_boundary_condition(
      std::pair<const dealii::types::boundary_id, const std::shared_ptr<dealii::Function<dim>>>
                         id_and_function,
      const std::string &type)
    {
      AssertThrow(!bc[type].contains(id_and_function.first), ExcBCAlreadyAssigned(type));

      bc[type][id_and_function.first] = id_and_function.second;
    }

    /**
     * @brief Set the time for all time-dependent boundary conditions.
     *
     * Updates the time for all boundary conditions that are associated with functions
     * supporting time-dependent behavior.
     *
     * @param time The current simulation time.
     */
    void
    set_time(const number time)
    {
      for (const auto &[type, outer_map] : bc)
        for (const auto &[id_, function_ptr] : outer_map)
          if (function_ptr)
            function_ptr->set_time(time);
    }

    /**
     * @brief Retrieve all boundary conditions of a specific type.
     *
     * Provides access to the boundary conditions associated with a given type.
     *
     * @param type The type of the boundary conditions to retrieve.
     * @param is_optional    A flag indicating whether the bc is optional.
     *                       Defaults to `true`, requiring the bc not to be present.
     * @return A map of boundary IDs to their corresponding functions.
     *
     * @throws dealii::ExcMessage If the specified type does not exist in the manager.
     */
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    get_bc_of_type(const std::string type, const bool is_optional = true) const
    {
      AssertThrow(is_optional || bc.contains(type), dealii::ExcMessage("Type not found: " + type));

      if (!bc.contains(type))
        return std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>();
      else
        return bc.at(type);
    }

    /**
     * @brief Get all boundary IDs associated with a specific type.
     *
     * Retrieves a list of boundary IDs for a given type of boundary condition.
     *
     * @param type The type of the boundary conditions.
     * @param is_optional    A flag indicating whether the bc is optional.
     *                       Defaults to `true`, requiring the bc not to be present.
     * @return A vector of boundary IDs (`dealii::types::boundary_id`) associated with the given type.
     */
    std::vector<dealii::types::boundary_id>
    get_indices_of_type(const std::string type, const bool is_optional = true) const
    {
      auto bc_of_type = get_bc_of_type(type, is_optional);
      auto kv         = std::views::keys(bc_of_type);
      return {kv.begin(), kv.end()};
    }

    /**
     * @brief Retrieve the type of a boundary condition by its ID.
     *
     * This function searches through the boundary condition manager to find a boundary
     * condition matching the specified ID and returns its associated type. If the ID
     * is not found, the behavior depends on the `is_optional` flag:
     * - If `is_optional` is `true`, the function returns an empty string.
     * - If `is_optional` is `false`, the function throws an exception.
     *
     * @param id The boundary ID to search for.
     * @param is_optional A flag indicating whether the boundary condition ID is optional.
     *                    - If `true`, the function returns an empty string if the ID is not found.
     *                    - If `false`, the function throws an exception if the ID is not found.
     */
    std::string
    get_type(dealii::types::boundary_id id, const bool is_optional = true) const
    {
      for (const auto &[type_, inner_map] : bc)
        if (inner_map.contains(id))
          return type_; // Found boundary ID, return type

      // If not found, check the is_optional flag
      AssertThrow(is_optional, dealii::ExcMessage("Boundary ID not found in boundary conditions."));

      // If optional, return an empty string as a fallback
      return "";
    }
  };
} // namespace MeltPoolDG
