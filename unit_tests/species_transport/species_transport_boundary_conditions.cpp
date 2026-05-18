#include <gtest/gtest.h>

#include <meltpooldg/species_transport/boundary_conditions.hpp>

#include <array>

namespace
{
  template <std::size_t n_partial_densities>
  struct MockDofView
  {
    mutable std::array<double, n_partial_densities> values{};
    mutable std::array<double, n_partial_densities> gradients{};

    double &
    partial_density(unsigned int i) const
    {
      return values[i];
    }

    double &
    grad_partial_density(unsigned int i) const
    {
      return gradients[i];
    }
  };

  TEST(SpeciesTransportBoundaryConditions, DirichletBoundary)
  {
    constexpr std::size_t n_partial_densities = 3;

    MockDofView<n_partial_densities> w_m;
    MockDofView<n_partial_densities> w_p;

    // Initialize inner values (w_m)
    for (unsigned int i = 0; i < n_partial_densities; ++i)
      {
        w_m.values[i]    = i + 1.0;
        w_m.gradients[i] = (i + 1.0) * 10.0;
      }

    std::array<double, n_partial_densities> dirichlet_values = {{100.0, 200.0, 300.0}};

    set_boundary_value_and_gradient<n_partial_densities>(
      w_p, w_m, MeltPoolDG::SpeciesTransport::BoundaryConditionType::dirichlet, dirichlet_values);

    // Check values -> should match Dirichlet input
    for (unsigned int i = 0; i < n_partial_densities; ++i)
      EXPECT_DOUBLE_EQ(w_p.values[i], dirichlet_values[i]);

    // Check gradients -> copied from w_m
    for (unsigned int i = 0; i < n_partial_densities; ++i)
      EXPECT_DOUBLE_EQ(w_p.gradients[i], w_m.gradients[i]);
  }

  TEST(SpeciesTransportBoundaryConditions, NeumannBoundary)
  {
    constexpr std::size_t n_partial_densities = 3;

    MockDofView<n_partial_densities> w_m;
    MockDofView<n_partial_densities> w_p;

    for (unsigned int i = 0; i < n_partial_densities; ++i)
      {
        w_m.values[i]    = i + 1.0;
        w_m.gradients[i] = (i + 1.0) * 10.0;
      }

    set_boundary_value_and_gradient<n_partial_densities, double>(
      w_p, w_m, MeltPoolDG::SpeciesTransport::BoundaryConditionType::neumann);

    // Values copied
    for (unsigned int i = 0; i < n_partial_densities; ++i)
      EXPECT_DOUBLE_EQ(w_p.values[i], w_m.values[i]);

    // Gradients flipped sign
    for (unsigned int i = 0; i < n_partial_densities; ++i)
      EXPECT_DOUBLE_EQ(w_p.gradients[i], -w_m.gradients[i]);
  }

  TEST(SpeciesTransportBoundaryConditions, WrongBoundaryType)
  {
    constexpr std::size_t n_partial_densities = 3;

    MockDofView<n_partial_densities> w_m;
    MockDofView<n_partial_densities> w_p;

    EXPECT_ANY_THROW((set_boundary_value_and_gradient<n_partial_densities, double>(
      w_p, w_m, static_cast<MeltPoolDG::SpeciesTransport::BoundaryConditionType>(999))));
  }

} // namespace
