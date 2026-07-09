#pragma once

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/compressible_flow/state_views_multi_species.hpp>

/**
 * This file contains type aliases for the state views used in the multi- and single-species
 * compressible flow solver. The main purpose is that the type aliases automatically resolve to the
 * correct state view type depending on the number of species in the simulation. For example, for
 * single-species simulations, the state views resolve to the standard DofStateView and
 * DofValueAndGradientStateView, while for multi-species simulations, they resolve to the
 * corresponding MultiSpeciesDofStateView and MultiSpeciesDofValueAndGradientStateView.
 */
namespace MeltPoolDG::CompressibleFlow
{
  /**
   * Type alias for the DofValueView for a specific number of species. This alias resolves to the
   * appropriate state view type depending on the number of species in the simulation.
   */
  template <int dim, int n_species, IsConservedStateCompatible<dim> StateType>
  using NSpeciesDofValueView =
    std::conditional_t<n_species == 1,
                       DofValueView<dim, StateType>,
                       MultiSpeciesDofValueView<dim, n_species, StateType>>;

  /**
   * Type alias for the DofStateView for a specific number of species. This alias resolves to the
   * appropriate state view type depending on the number of species in the simulation.
   */
  template <int dim, int n_species, typename number, IsConservedStateCompatible<dim> StateType>
  using NSpeciesDofStateView =
    std::conditional_t<n_species == 1,
                       DofStateView<dim, number, StateType>,
                       MultiSpeciesDofStateView<dim, n_species, number, StateType>>;

  /**
   * Type alias for the DofValueAndGradientStateView for a specific number of species. This alias
   * resolves to the appropriate state view type depending on the number of species in the
   * simulation.
   */
  template <int dim,
            int n_species,
            typename number,
            IsConservedStateCompatible<dim>    Value,
            IsConservedGradientCompatible<dim> Gradient>
  using NSpeciesDofValueAndGradientStateView = std::conditional_t<
    n_species == 1,
    DofValueAndGradientStateView<dim, number, Value, Gradient>,
    MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>;

  /**
   * Type alias for the FluxView for a specific number of species. This alias resolves to the
   * appropriate flux view type depending on the number of species in the simulation.
   */
  template <int dim, int n_species, typename FluxType>
  using NSpeciesFluxView = std::conditional_t<n_species == 1,
                                              FluxView<dim, FluxType>,
                                              MultiSpeciesFluxView<dim, n_species, FluxType>>;

} // namespace MeltPoolDG::CompressibleFlow
