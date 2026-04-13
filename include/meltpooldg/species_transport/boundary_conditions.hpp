#pragma once

#include <deal.II/base/exceptions.h>

#include <array>

namespace MeltPoolDG::SpeciesTransport
{

  /**
   * Enum class for specifying the type of boundary condition for partial densities in species
   * transport.
   */
  enum class BoundaryConditionType
  {
    dirichlet,
    neumann
  };

  /**
   * Sets the boundary value and gradient for partial densities in species transport. The boundary
   * condition is only applied to all partial densities and their gradients.
   *
   * @param w_p The writable degree of freedom view for the values on the outer face.
   * @param w_m The read-only degree of freedom view for the values on the inner face.
   * @param boundary_type The type of boundary condition to apply.
   * @param dirichlet_boundary_values The prescribed partial densities for Dirichlet boundary
   * conditions (optional, only used if boundary_type is dirichlet).
   */
  template <std::size_t n_partial_densities,
            typename VectorizedArrayType,
            typename ReadOnlyDofView,
            typename WritableDofView>
  void
  set_boundary_value_and_gradient(
    const WritableDofView                                      &w_p,
    const ReadOnlyDofView                                      &w_m,
    const BoundaryConditionType                                 boundary_type,
    const std::array<VectorizedArrayType, n_partial_densities> &dirichlet_boundary_values = {})
  {
    const auto set_to_value = [](const ReadOnlyDofView     &src,
                                 const WritableDofView     &dst,
                                 const VectorizedArrayType &scaling_factor = 1.) {
      for (unsigned int species = 0; species < n_partial_densities; ++species)
        {
          dst.partial_density(species) = src.partial_density(species) * scaling_factor;
        }
    };

    const auto set_to_gradient = [](const ReadOnlyDofView     &src,
                                    const WritableDofView     &dst,
                                    const VectorizedArrayType &scaling_factor = 1.) {
      for (unsigned int species = 0; species < n_partial_densities; ++species)
        dst.grad_partial_density(species) = src.grad_partial_density(species) * scaling_factor;
    };

    if (boundary_type == BoundaryConditionType::dirichlet)
      {
        for (unsigned int species = 0; species < n_partial_densities; ++species)
          w_p.partial_density(species) = dirichlet_boundary_values[species];

        set_to_gradient(w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::neumann)
      {
        set_to_value(w_m, w_p);
        set_to_gradient(w_m, w_p, -1.);
      }
    else
      {
        AssertThrow(false, dealii::ExcMessage("Unknown boundary type for species transport."));
      }
  }
} // namespace MeltPoolDG::SpeciesTransport
