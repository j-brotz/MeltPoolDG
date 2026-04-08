#pragma once

#include <meltpooldg/compressible_flow/kernels.hpp>
#include <meltpooldg/compressible_flow/kernels_multi_species.hpp>

/**
 * This file contains type aliases for the kernels used in the multi- and single-species
 * compressible flow solver. The main purpose is that the type aliases automatically resolve to the
 * correct kernel type depending on the number of species in the simulation. For example, for
 * single-species simulations, the kernels resolve to the standard ConvectiveFlux and
 * DiffusiveFlux, while for multi-species simulations, they resolve to the
 * corresponding SpeciesTransportConvectiveFlux and SpeciesTransportDiffusiveFlux.
 */
namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, int n_species, typename number, typename Value, typename Flux>
  struct ConcreteConvectiveFluxImpl
  {
    using type = SpeciesTransportConvectiveFlux<dim, n_species, number, Value, Flux>;
  };

  template <int dim, typename number, typename Value, typename Flux>
  struct ConcreteConvectiveFluxImpl<dim, 1, number, Value, Flux>
  {
    using type = ConvectiveFlux<dim, number, Value, Flux>;
  };

  /**
   * Type alias for the ConvectiveFlux for a specific number of species. This alias resolves to the
   * appropriate kernel type depending on the number of species in the simulation.
   */
  template <int dim, int n_species, typename number, typename Value, typename Flux>
  using NSpeciesConvectiveFlux =
    typename ConcreteConvectiveFluxImpl<dim, n_species, number, Value, Flux>::type;

  template <int dim,
            int n_species,
            typename number,
            typename Value,
            typename Gradient,
            typename Flux>
  struct ConcreteDiffusiveFluxImpl
  {
    using type = SpeciesTransportDiffusiveFlux<dim, n_species, number, Value, Gradient, Flux>;
  };

  template <int dim, typename number, typename Value, typename Gradient, typename Flux>
  struct ConcreteDiffusiveFluxImpl<dim, 1, number, Value, Gradient, Flux>
  {
    using type = DiffusiveFlux<dim, number, Value, Gradient, Flux>;
  };

  /**
   * Type alias for the DiffusiveFlux for a specific number of species. This alias resolves to the
   * appropriate kernel type depending on the number of species in the simulation.
   */
  template <int dim,
            int n_species,
            typename number,
            typename Value,
            typename Gradient,
            typename Flux>
  using NSpeciesDiffusiveFlux =
    typename ConcreteDiffusiveFluxImpl<dim, n_species, number, Value, Gradient, Flux>::type;

} // namespace MeltPoolDG::CompressibleFlow
