#include <gtest/gtest.h>

#include <meltpooldg/species_transport/effective_material_mixin.hpp>

#include <array>
#include <vector>

namespace
{
  using namespace MeltPoolDG::SpeciesTransport;

  constexpr int n_species = 3;

  struct MockMaterial : EffectiveMaterialMixin<n_species, double, double, MockMaterial>
  {
    // Sample data
    std::array<double, n_species>                        mass_frac      = {{0.2, 0.3, 0.5}};
    std::array<double, n_species>                        molar_mass_val = {{10.0, 20.0, 30.0}};
    std::array<std::array<double, n_species>, n_species> diff_coeff     = {
      {{{0., 1., 2.}}, {{1., 0., 3.}}, {{2., 3., 0.}}}};

    double
    mass_fraction(unsigned i) const
    {
      return mass_frac[i];
    }
    double
    molar_mass(unsigned i) const
    {
      return molar_mass_val[i];
    }

    double
    binary_diffusion_coefficient(unsigned i, unsigned j) const
    {
      return diff_coeff[i][j];
    }
  };

  TEST(EffectiveMaterialMixinTest, MoleFractionNormal)
  {
    MockMaterial material;

    std::array<double, n_species> expected = {{12. / 31., 9. / 31., 10. / 31.}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(material.mole_fraction(i), expected[i]);
      }
  }

  TEST(EffectiveMaterialMixinTest, MoleFractionSingleSpeciesDominates)
  {
    MockMaterial material;
    material.mass_frac = {{1.0, 0.0, 0.0}};

    std::array<double, n_species> expected = {{1.0, 0.0, 0.0}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(material.mole_fraction(i), expected[i]);
      }
  }

  TEST(EffectiveMaterialMixin, MixtureAveragedDiffusionBasic)
  {
    // Normal case: all binary diffusion coefficients are positive, and all species contribute to
    // the mixture diffusion coefficient.

    MockMaterial m;

    std::array<double, n_species> expected = {
      {1.771428571428572, 1.415217391304348, 1.722222222222222}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(m.mixture_averaged_diffusion_coefficient(i), expected[i]);
      }
  }


  TEST(EffectiveMaterialMixin, MixtureDiffusionAllZero)
  {
    // All binary diffusion coefficients are zero. Therefore, mixture-averaged diffusion coefficient
    // should be zero.

    MockMaterial m;

    // Set all binary diffusion coefficients to zero
    for (auto &row : m.diff_coeff)
      for (auto &val : row)
        val = 0.0;

    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(m.mixture_averaged_diffusion_coefficient(i), 0.0);
      }
  }

  TEST(EffectiveMaterialMixin, MixtureDiffusionSinglePair)
  {
    // Binary diffusion coefficient is only non-zero for one pair of species. The mixture-averaged
    // diffusion coefficient should be equal to the binary diffusion coefficient of that pair.

    MockMaterial m;

    m.diff_coeff = {{{{0., 0.5, 0.}}, {{0.5, 0., 0.}}, {{0., 0., 0.}}}};

    std::array<double, n_species> expected = {{1.377777777777778, 0.9041666666666666, 0.0}};

    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(m.mixture_averaged_diffusion_coefficient(i), expected[i]);
      }
  }

  TEST(EffectiveMaterialMixin, MixtureDiffusionPureSpecies)
  {
    // Only one species is present (mass fraction of 1)

    MockMaterial m;
    m.mass_frac = {{1.0, 0.0, 0.0}};

    std::array<double, n_species> expected = {{0.0, 1.0, 2.0}};

    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(m.mixture_averaged_diffusion_coefficient(i), expected[i]);
      }
  }

  TEST(EffectiveMaterialMixinTest, EffectiveMaterialProperty)
  {
    struct SpeciesMaterial
    {
      double prop;
    };

    MockMaterial                 material;
    std::vector<SpeciesMaterial> species = {{10.0}, {20.0}, {30.0}};

    double result = material.effective_material_property(species, &SpeciesMaterial::prop);

    // Expected = sum_i mass_fraction(i) * prop_i
    EXPECT_DOUBLE_EQ(result, 23.);
  }
} // namespace
