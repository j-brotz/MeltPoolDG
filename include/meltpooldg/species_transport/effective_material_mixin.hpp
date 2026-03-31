#pragma once

#include <deal.II/base/vectorization.h>

#include <boost/range/adaptor/indexed.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace MeltPoolDG::SpeciesTransport
{
  template <int n_species, typename number, typename ValueType, typename Derived>
  struct EffectiveMaterialMixin
  {
    /**
     * Return the mole fraction of a species component computed from the mass fractions and molar
     * masses of all species.
     *
     * @param species_component The index of the species component of interest.
     */
    ValueType
    mole_fraction(const unsigned species_component) const
      requires requires() {
        {
          std::declval<const Derived>().mass_fraction(species_component)
        } -> std::convertible_to<ValueType>;
        {
          std::declval<const Derived>().molar_mass(species_component)
        } -> std::convertible_to<ValueType>;
      }
    {
      ValueType total_amount_of_substance = 0.;
      for (unsigned int i = 0; i < n_species; ++i)
        {
          total_amount_of_substance += derived().mass_fraction(i) / derived().molar_mass(i);
        }

      ValueType species_amount_of_substance =
        derived().mass_fraction(species_component) / derived().molar_mass(species_component);
      return species_amount_of_substance / total_amount_of_substance;
    }

    /**
     * Return the mixture-averaged diffusion coefficient for a species component. The
     * mixture-averaged diffusion coefficient is computed from the binary diffusion coefficients of
     * the species pairs and the mass as well as mole fractions of all species according to
     *
     * \f[
     * D_i = \frac{1 - Y_i}{\sum_{j \neq i} \frac{X_j}{D_{ij}}}]
     * \f]
     *
     * with the species mass fraction \f$Y_i\f$, mole fraction \f$X_j\f$, and binary diffusion
     * coefficient \f$D_{ij}\f$.
     *
     * @param species_component The index of the species component of interest.
     */
    ValueType
    mixture_averaged_diffusion_coefficient(const unsigned int species_component) const
      requires requires() {
        {
          std::declval<const Derived>().binary_diffusion_coefficient(species_component,
                                                                     species_component)
        } -> std::convertible_to<ValueType>;
        {
          std::declval<const Derived>().mass_fraction(species_component)
        } -> std::convertible_to<ValueType>;
      }
    {
      ValueType denominator = 0.;
      for (unsigned int i = 0; i < n_species; ++i)
        {
          // If a single binary diffusion coefficient is zero, we simply do not take into account
          // the diffusion of the respective species pair.
          if (i != species_component and
              derived().binary_diffusion_coefficient(species_component, i) > 0.)
            {
              denominator +=
                mole_fraction(i) / derived().binary_diffusion_coefficient(species_component, i);
            }
        }

      const auto compute_final_mixture_averaged_diffusion_coefficient =
        [&](const ValueType &denominator) -> ValueType {
        return (1. - derived().mass_fraction(species_component)) / denominator;
      };

      // If the denominator is zero, it means that the species does not diffuse with any other
      // available species. In this case, we set the mixture-averaged diffusion coefficient to zero.
      constexpr ValueType tolerance = ValueType(1e-20);
      if constexpr (std::is_floating_point_v<ValueType>)
        {
          return denominator < tolerance ?
                   ValueType(0.) :
                   compute_final_mixture_averaged_diffusion_coefficient(denominator);
        }
      else
        {
          return dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
            denominator,
            tolerance,
            compute_final_mixture_averaged_diffusion_coefficient(denominator),
            ValueType(0.));
        }
    }

    /**
     * Compute the effective material property for a mixture of species.
     *
     * @param species_material A container of species materials holding objects if type SpeciesMaterial.
     * @param property A pointer to the material property to be computed.
     * @return The effective material property for the mixture.
     */
    template <typename ContainerType, typename SpeciesMaterial>
    ValueType
    effective_material_property(const ContainerType &species_material,
                                number SpeciesMaterial::*property) const
      requires requires(int i) {
        {
          std::declval<const Derived>().mass_fraction(i)
        } -> std::convertible_to<ValueType>;
      }
    {
      ValueType result = 0.;
      for (const auto [index, species] : species_material | boost::adaptors::indexed())
        {
          if (index >= n_species)
            break;
          result += derived().mass_fraction(index) * (species.*property);
        }
      return result;
    }

  private:
    decltype(auto)
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }
  };
} // namespace MeltPoolDG::SpeciesTransport
