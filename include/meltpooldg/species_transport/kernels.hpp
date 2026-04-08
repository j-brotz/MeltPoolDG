#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/vectorization.h>

namespace MeltPoolDG::SpeciesTransport
{
  /**
   * @brief Computes the convective species transport flux for each species.
   *
   * Updates the species partial density flux based on the conserved variables and their velocity.
   * Only iterates over n_species-1 since the last species is assumed to be not present in the
   * solution vector but computed from the mass fractions summing to one.
   *
   * @param conserved_variables The conserved variables of the system.
   * @param flux The flux object to be updated.
   */
  template <int n_species, typename ConservedVariablesView, typename WritableFluxView>
  inline DEAL_II_ALWAYS_INLINE //
    void
    compute_convective_flux(const ConservedVariablesView &conserved_variables,
                            const WritableFluxView       &flux)
  {
    for (unsigned int i = 0; i < n_species - 1; ++i)
      {
        flux.partial_density_flux(i) +=
          conserved_variables.partial_density(i) * conserved_variables.velocity();
      }
  }

  /**
   * Computes an approximation of the Fickian diffusion flux for a single species as presented in
   *
   * Cook et al., "Enthalpy diffusion in multicomponent flows", Physics of Fluids 21, 055109(2009)
   *
   * @param conserved_variables The conserved variables of the system including their gradients.
   * @param species_idx Index of the species for which to compute the flux.
   *
   * @return The approximate Fickian diffusion flux for the given species.
   */
  template <int n_species, typename ConservedVariablesView>
  [[nodiscard]] inline DEAL_II_ALWAYS_INLINE //
    auto
    fickian_diffusion_approximation(const ConservedVariablesView &conserved_variables,
                                    const unsigned int            species_idx)
  {
    auto flux = conserved_variables.mixture_averaged_diffusion_coefficient(species_idx) *
                conserved_variables.grad_mass_fraction(species_idx);

    for (unsigned int i = 0; i < n_species; ++i)
      flux -= conserved_variables.mass_fraction(species_idx) *
              conserved_variables.mixture_averaged_diffusion_coefficient(i) *
              conserved_variables.grad_mass_fraction(i);

    flux *= -conserved_variables.density();
    return flux;
  }

  /**
   * Computes the diffusive species transport flux for each species based on the Fickian diffusion
   * approximation. Only iterates over n_species-1 since the last species is assumed to be not
   * present in the solution vector but computed from the mass fractions summing to one.
   *
   * @param conserved_variables The conserved variables of the system including their gradients.
   * @param flux The flux object to be updated.
   */
  template <int n_species, typename ConservedVariablesView, typename WritableFluxView>
  inline DEAL_II_ALWAYS_INLINE //
    void
    compute_diffusive_flux(const ConservedVariablesView &conserved_variables,
                           const WritableFluxView       &flux)
  {
    for (unsigned int i = 0; i < n_species - 1; ++i)
      {
        flux.partial_density_flux(i) -=
          fickian_diffusion_approximation<n_species, ConservedVariablesView>(conserved_variables,
                                                                             i);
      }
  }

  /**
   * Computes the interdiffusional enthalpy flux based on the Fickian diffusion approximation for
   * each species. The mathematical formulation of the interdiffusional enthalpy flux is given in
   *
   * Cook et al., "Enthalpy diffusion in multicomponent flows", Physics of Fluids 21, 055109(2009)
   *
   * @param conserved_variables The conserved variables of the system including their gradients.
   * @param flux The flux object to be updated with the interdiffusional enthalpy flux contribution
   * to the energy flux.
   */
  template <int n_species,
            typename VectorizedArrayType,
            typename ConservedVariablesView,
            typename WritableFluxView>
  void
  interdiffusional_enthalpy_flux(const ConservedVariablesView &conserved_variables,
                                 const WritableFluxView       &flux)
  {
    const auto species_enthalpy = [&](const int idx) -> VectorizedArrayType {
      // If the partial density of the species is zero, set the species enthalpy to zero to avoid
      // NaNs in the interdiffusional enthalpy flux.
      return dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
        conserved_variables.mass_fraction(idx),
        VectorizedArrayType(1e-15),
        conserved_variables.partial_inner_energy(idx) +
          conserved_variables.partial_pressure(idx) / conserved_variables.partial_density(idx),
        VectorizedArrayType(0.));
    };

    for (unsigned int i = 0; i < n_species; ++i)
      flux.energy_flux() +=
        species_enthalpy(i) *
        fickian_diffusion_approximation<n_species, ConservedVariablesView>(conserved_variables, i);
  }
} // namespace MeltPoolDG::SpeciesTransport
