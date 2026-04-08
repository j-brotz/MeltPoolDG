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
  template <int dim, int n_species, IsConservedStateCompatible<dim> StateType>
  struct ConcreteDofValueViewImpl
  {
    using type = MultiSpeciesDofValueView<dim, n_species, StateType>;
  };

  template <int dim, IsConservedStateCompatible<dim> StateType>
  struct ConcreteDofValueViewImpl<dim, 1, StateType>
  {
    using type = DofValueView<dim, StateType>;
  };

  /**
   * Type alias for the DofValueView for a specific number of species. This alias resolves to the
   * appropriate state view type depending on the number of species in the simulation.
   */
  template <int dim, int n_species, IsConservedStateCompatible<dim> StateType>
  using NSpeciesDofValueView = typename ConcreteDofValueViewImpl<dim, n_species, StateType>::type;

  template <int dim, int n_species, typename number, IsConservedStateCompatible<dim> StateType>
  struct ConcreteDofStateViewImpl
  {
    using type = MultiSpeciesDofStateView<dim, n_species, number, StateType>;
  };

  template <int dim, typename number, IsConservedStateCompatible<dim> StateType>
  struct ConcreteDofStateViewImpl<dim, 1, number, StateType>
  {
    using type = DofStateView<dim, number, StateType>;
  };

  /**
   * Type alias for the DofStateView for a specific number of species. This alias resolves to the
   * appropriate state view type depending on the number of species in the simulation.
   */
  template <int dim, int n_species, typename number, IsConservedStateCompatible<dim> StateType>
  using NSpeciesDofStateView =
    typename ConcreteDofStateViewImpl<dim, n_species, number, StateType>::type;

  template <int dim,
            int n_species,
            typename number,
            IsConservedStateCompatible<dim>    Value,
            IsConservedGradientCompatible<dim> Gradient>
  struct ConcreteDofValueAndGradientStateViewImpl
  {
    using type = MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>;
  };

  template <int dim,
            typename number,
            IsConservedStateCompatible<dim>    Value,
            IsConservedGradientCompatible<dim> Gradient>
  struct ConcreteDofValueAndGradientStateViewImpl<dim, 1, number, Value, Gradient>
  {
    using type = DofValueAndGradientStateView<dim, number, Value, Gradient>;
  };

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
  using NSpeciesDofValueAndGradientStateView =
    typename ConcreteDofValueAndGradientStateViewImpl<dim, n_species, number, Value, Gradient>::
      type;

  template <int dim, int n_species, typename FluxType>
  struct ConcreteFluxViewImpl
  {
    using type = MultiSpeciesFluxView<dim, n_species, FluxType>;
  };

  template <int dim, typename FluxType>
  struct ConcreteFluxViewImpl<dim, 1, FluxType>
  {
    using type = FluxView<dim, FluxType>;
  };

  /**
   * Type alias for the FluxView for a specific number of species. This alias resolves to the
   * appropriate flux view type depending on the number of species in the simulation.
   */
  template <int dim, int n_species, typename FluxType>
  using NSpeciesFluxView = typename ConcreteFluxViewImpl<dim, n_species, FluxType>::type;

} // namespace MeltPoolDG::CompressibleFlow
