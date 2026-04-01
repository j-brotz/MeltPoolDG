#include <gtest/gtest.h>

#include <meltpooldg/species_transport/kernels.hpp>

#include <array>

namespace
{
  using namespace MeltPoolDG::SpeciesTransport;

  /**
   * Mock view class to test the convective and diffusive flux kernels. It provides read-only access
   * to the (dummy) conserved variables and derived quantities.
   */
  template <int n_species>
  struct MockView
  {
    std::array<double, n_species> partial_densities;
    std::array<double, n_species> grad_mass_fractions;
    std::array<double, n_species> mass_fractions;
    std::array<double, n_species> mixture_diff_coeffs;
    std::array<double, n_species> partial_energies;
    std::array<double, n_species> partial_pressures;
    double                        vel = 3.0;
    double                        rho = 2.0;

    double
    partial_density(unsigned int i) const
    {
      return partial_densities[i];
    }

    double
    mass_fraction(unsigned int i) const
    {
      return mass_fractions[i];
    }

    double
    mixture_averaged_diffusion_coefficient(unsigned int i) const
    {
      return mixture_diff_coeffs[i];
    }

    double
    grad_mass_fraction(unsigned int i) const
    {
      return grad_mass_fractions[i];
    }

    double
    partial_inner_energy(unsigned int i) const
    {
      return partial_energies[i];
    }

    double
    partial_pressure(unsigned int i) const
    {
      return partial_pressures[i];
    }

    double
    velocity() const
    {
      return vel;
    }

    double
    density() const
    {
      return rho;
    }
  };

  /**
   * Mock flux class to test the convective and diffusive flux kernels. It provides mutable access
   * to the partial density fluxes and energy flux.
   */
  template <int n_species>
  struct MockFlux
  {
    mutable std::array<double, n_species> partial_density_fluxes = {};
    mutable double                        energy_flux_val        = 4.0;

    double &
    partial_density_flux(unsigned int i) const
    {
      return partial_density_fluxes[i];
    }

    double &
    energy_flux() const
    {
      return energy_flux_val;
    }
  };


  TEST(ConvectiveFluxTest, NSpecies)
  {
    // This test checks that the convective flux is computed correctly for a case with three
    // species, and that the last species is correctly skipped in the loop since it is not part of
    // the conserved variables.

    constexpr int n_species = 3;

    MockView<n_species> conserved_variables;
    conserved_variables.partial_densities = {{2.0, 3.0, 4.0}};
    conserved_variables.vel               = 2.0;

    MockFlux<n_species> flux;
    flux.partial_density_fluxes = {
      {1.0, 2.0, 0.0}}; // to check if convective flux modifies them as expected

    convective_flux<n_species>(conserved_variables, flux);

    // The last species should be skipped as it is not part of the conserved variables
    std::array<double, n_species> expected_fluxes = {{5.0, 8.0, 0.0}};

    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_NEAR(flux.partial_density_flux(i), expected_fluxes[i], 1e-15);
      }
  }

  TEST(ConvectiveFluxTest, SingleSpecies)
  {
    // This test verifies that if there is only one species, the convective flux loop does not run
    // and the flux remains zero.

    constexpr int n_species = 1;

    MockView<n_species> conserved_variables;
    MockFlux<n_species> flux;

    convective_flux<n_species>(conserved_variables, flux);
    flux.partial_density_fluxes = {{1.0}}; // to check if convective flux modifies them as expected

    // With n_species=1, loop should not run, flux remains 0
    EXPECT_NEAR(flux.partial_density_flux(0), 1.0, 1e-15);
  }

  TEST(DiffusiveFluxTest, NSpecies)
  {
    // This test checks that the diffusive flux is computed correctly for a case with three species,
    // and that the last species is correctly skipped in the loop since it is not part of the
    // conserved variables.

    constexpr int n_species = 3;

    MockView<n_species> conserved_variables;
    conserved_variables.partial_densities   = {{2.0, 3.0, 4.0}};
    conserved_variables.mixture_diff_coeffs = {{0.01, 0.02, 0.03}};
    conserved_variables.grad_mass_fractions = {{0.1, -0.1, 0.0}};
    conserved_variables.mass_fractions      = {{0.4, 0.25, 0.35}};
    conserved_variables.rho                 = 3.0;
    conserved_variables.vel                 = 2.0;

    MockFlux<n_species> flux;
    flux.partial_density_fluxes = {
      {1.0, -2.0, 0.0}}; // to check if convective flux modifies them as expected

    diffusive_flux<n_species>(conserved_variables, flux);

    // The last species should be skipped as it is not part of the conserved variables
    std::array<double, n_species> expected_fluxes = {{1.0042, -2.00525, 0.0}};

    for (unsigned int i = 0; i < n_species - 1; ++i)
      {
        EXPECT_NEAR(flux.partial_density_flux(i), expected_fluxes[i], 1e-15);
      }
  }

  TEST(FickianDiffusionApproximationTest, NSpecies)
  {
    // This test verifies that the Fickian diffusion approximation is computed correctly for a case
    // with two species, and that the contributions from both the direct diffusion term and the
    // correction term are included in the result.

    constexpr int n_species = 2;

    MockView<n_species> conserved_variables;
    conserved_variables.mixture_diff_coeffs = {{0.01, 0.02}};
    conserved_variables.grad_mass_fractions = {{0.1, -0.1}};
    conserved_variables.mass_fractions      = {{0.4, 0.6}};
    conserved_variables.rho                 = 3.0;

    std::array<double, n_species> expected_diffusion_approximations = {{-0.0042, 0.0042}};
    EXPECT_NEAR(fickian_diffusion_approximation<n_species>(conserved_variables, 0),
                expected_diffusion_approximations[0],
                1e-15);
    EXPECT_NEAR(fickian_diffusion_approximation<n_species>(conserved_variables, 1),
                expected_diffusion_approximations[1],
                1e-15);
  }

  TEST(FickianDiffusionApproximationTest, ZeroSpeciesGradient)
  {
    // This test checks that if the gradients of mass fractions are zero, the Fickian diffusion
    // approximation returns zero flux, even if the diffusion coefficients and mass fractions are
    // non-zero, and that no NaNs are produced in the calculation.

    constexpr int n_species = 2;

    MockView<n_species> conserved_variables;
    conserved_variables.mixture_diff_coeffs = {{0.01, 0.02}};
    conserved_variables.grad_mass_fractions = {{0.0, -0.0}};
    conserved_variables.mass_fractions      = {{0.4, 0.6}};
    conserved_variables.rho                 = 3.0;

    EXPECT_NEAR(fickian_diffusion_approximation<n_species>(conserved_variables, 0), 0.0, 1e-15);
    EXPECT_NEAR(fickian_diffusion_approximation<n_species>(conserved_variables, 1), 0.0, 1e-15);
  }

  TEST(InterdiffusionalEnthalpyFluxTest, AllPartialDensitiesNotZero)
  {
    // This test verifies that the interdiffusional enthalpy flux is computed correctly when all
    // partial densities are non-zero.

    constexpr int n_species = 2;

    MockView<n_species> conserved_variables;
    conserved_variables.partial_densities   = {{2.0, 3.0}};
    conserved_variables.mixture_diff_coeffs = {{0.01, 0.02}};
    conserved_variables.grad_mass_fractions = {{0.1, -0.1}};
    conserved_variables.mass_fractions      = {{0.4, 0.6}};
    conserved_variables.partial_energies    = {{10.0, 20.0}};
    conserved_variables.partial_pressures   = {{2.0, 3.0}};
    conserved_variables.rho                 = 3.0;
    conserved_variables.vel                 = 2.0;

    MockFlux<n_species> flux;

    interdiffusional_enthalpy_flux<2, double>(conserved_variables, flux);

    EXPECT_NEAR(flux.energy_flux(), 4.042, 1e-15);
  }

  TEST(InterdiffusionalEnthalpyFluxTest, OnePartialDensityZero)
  {
    // This test checks that if one of the species has a zero partial density, its contribution to
    // the interdiffusional enthalpy flux is zero, and that the other species' contribution is still
    // correctly computed without NaNs affecting the result.

    constexpr int n_species = 2;

    MockView<n_species> conserved_variables;
    conserved_variables.partial_densities   = {{2.0, 0.0}};
    conserved_variables.mixture_diff_coeffs = {{0.01, 0.02}};
    conserved_variables.grad_mass_fractions = {{0.1, -0.1}};
    conserved_variables.mass_fractions      = {{1.0, 0.0}};
    conserved_variables.partial_energies    = {{10.0, 20.0}};
    conserved_variables.partial_pressures   = {{2.0, 3.0}};
    conserved_variables.rho                 = 3.0;
    conserved_variables.vel                 = 2.0;


    MockFlux<n_species> flux;

    interdiffusional_enthalpy_flux<2, double>(conserved_variables, flux);

    EXPECT_NEAR(flux.energy_flux(), 3.934, 1e-15);
  }

} // namespace
