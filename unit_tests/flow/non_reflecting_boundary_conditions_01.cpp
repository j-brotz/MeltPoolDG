/**
 * @file non_reflecting_boundary_conditions_01.cpp
 * @brief This file collects unit tests of the pointwise kernels of the polynomial-correction
 * Navier-Stokes characteristic boundary conditions (PC-NSCBC) for single- and multi-species
 * compressible flows.
 */

#include <gtest/gtest.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/non_reflecting_boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/state_views_n_species.hpp>

#include <cmath>
#include <vector>

namespace
{
  using namespace MeltPoolDG::CompressibleFlow;
  namespace NRBC = MeltPoolDG::CompressibleFlow::NonReflectingBoundary;

  using VectorizedArrayType = dealii::VectorizedArray<double>;

  constexpr double tolerance = 1e-12;

  /// Material of a single ideal gas.
  MaterialPhaseData<double>
  single_species_material()
  {
    MaterialPhaseData<double> material;
    material.eos_type                              = EquationOfState::ideal_gas;
    material.number_of_species                     = 1;
    material.gamma                                 = 1.4;
    material.specific_gas_constant                 = 287.;
    material.species_data[0].gamma                 = 1.4;
    material.species_data[0].specific_gas_constant = 287.;
    return material;
  }

  /// Material of a mixture of two ideal gases with different heat capacity ratios.
  MaterialPhaseData<double>
  two_species_material()
  {
    MaterialPhaseData<double> material;
    material.eos_type                              = EquationOfState::ideal_gas;
    material.number_of_species                     = 2;
    material.species_data[0].gamma                 = 1.4;
    material.species_data[0].specific_gas_constant = 287.;
    material.species_data[1].gamma                 = 1.2;
    material.species_data[1].specific_gas_constant = 200.;
    return material;
  }

  template <int n_species>
  MaterialPhaseData<double>
  material()
  {
    return n_species == 1 ? single_species_material() : two_species_material();
  }

  /// Compare all lanes of two vectorized arrays with a relative tolerance.
  void
  expect_near(const VectorizedArrayType &value,
              const VectorizedArrayType &expected,
              const double               scale = 1.)
  {
    for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
      EXPECT_NEAR(value[v], expected[v], tolerance * std::max(scale, std::abs(expected[v])));
  }

  template <int n, int dim>
  void
  expect_near(const dealii::Tensor<1, n, dealii::Tensor<1, dim, VectorizedArrayType>> &value,
              const dealii::Tensor<1, n, dealii::Tensor<1, dim, VectorizedArrayType>> &expected,
              const double                                                             scale = 1.)
  {
    for (unsigned int c = 0; c < n; ++c)
      for (unsigned int d = 0; d < dim; ++d)
        expect_near(value[c][d], expected[c][d], scale);
  }

  template <int n>
  void
  expect_near(const dealii::Tensor<1, n, VectorizedArrayType> &value,
              const dealii::Tensor<1, n, VectorizedArrayType> &expected,
              const double                                     scale = 1.)
  {
    for (unsigned int c = 0; c < n; ++c)
      expect_near(value[c], expected[c], scale);
  }

  /// Primitive state (rho, u, p, Y) of a subsonic flow.
  template <int dim, int n_species>
  NRBC::PrimitiveVariables<dim, double, n_species>
  reference_primitive_state(const dealii::Tensor<1, dim> &velocity)
  {
    using Idx = NRBC::PrimitiveIndex<dim>;
    NRBC::PrimitiveVariables<dim, double, n_species> q;
    q[Idx::density] = 1.2;
    for (unsigned int d = 0; d < dim; ++d)
      q[Idx::velocity + d] = velocity[d];
    q[Idx::pressure] = 1.;
    if constexpr (n_species > 1)
      q[Idx::mass_fraction] = 0.3;
    return q;
  }

  /// Gradient of a field which varies only in the direction of @p normal.
  template <int dim, int n_species>
  NRBC::PrimitiveGradient<dim, double, n_species>
  normal_gradient(const NRBC::PrimitiveVariables<dim, double, n_species> &normal_derivative,
                  const dealii::Tensor<1, dim, VectorizedArrayType>      &normal)
  {
    NRBC::PrimitiveGradient<dim, double, n_species> gradient;
    for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
      gradient[c] = normal_derivative[c] * normal;
    return gradient;
  }

  template <int dim, int n_species>
  VectorizedArrayType
  speed_of_sound(const NRBC::PrimitiveVariables<dim, double, n_species> &q,
                 const MaterialPhaseData<double>                        &material)
  {
    const auto w = NRBC::conserved_variables<dim, n_species, double>(q, material);
    return NSpeciesDofStateView<dim,
                                n_species,
                                double,
                                const ConservedVariablesType<dim, double, n_species>>(w, material)
      .speed_of_sound();
  }

  /**
   * Normal derivatives of the primitive variables of the characteristic waves, i.e. the columns
   * of the matrix S. The vorticity wave is given for the tangent @p tangent.
   */
  template <int dim, int n_species>
  struct CharacteristicWaves
  {
    using Primitive = NRBC::PrimitiveVariables<dim, double, n_species>;
    using Idx       = NRBC::PrimitiveIndex<dim>;

    CharacteristicWaves(const Primitive                                   &q,
                        const VectorizedArrayType                         &a,
                        const dealii::Tensor<1, dim, VectorizedArrayType> &normal,
                        const dealii::Tensor<1, dim, VectorizedArrayType> &tangent)
    {
      const VectorizedArrayType rho = q[Idx::density];

      acoustic_minus[Idx::density]  = rho / a;
      acoustic_plus[Idx::density]   = rho / a;
      acoustic_minus[Idx::pressure] = rho * a;
      acoustic_plus[Idx::pressure]  = rho * a;
      for (unsigned int d = 0; d < dim; ++d)
        {
          acoustic_minus[Idx::velocity + d] = -normal[d];
          acoustic_plus[Idx::velocity + d]  = normal[d];
          vorticity[Idx::velocity + d]      = tangent[d];
        }
      entropy[Idx::density] = 1.;
      if constexpr (n_species > 1)
        species[Idx::mass_fraction] = 1.;
    }

    /// wave with eigenvalue u_n - a
    Primitive acoustic_minus;
    /// wave with eigenvalue u_n
    Primitive entropy;
    /// wave with eigenvalue u_n
    Primitive vorticity;
    /// wave with eigenvalue u_n
    Primitive species;
    /// wave with eigenvalue u_n + a
    Primitive acoustic_plus;
  };

  template <int dim>
  dealii::Tensor<1, dim, VectorizedArrayType>
  to_vectorized(const dealii::Tensor<1, dim> &tensor)
  {
    dealii::Tensor<1, dim, VectorizedArrayType> result;
    for (unsigned int d = 0; d < dim; ++d)
      result[d] = tensor[d];
    return result;
  }

  // -----------------------------------------------------------------------------------------------
  // conversion between primitive and conserved variables
  // -----------------------------------------------------------------------------------------------

  template <int n_species>
  void
  check_primitive_conserved_round_trip()
  {
    constexpr int dim       = 3;
    constexpr int n_vars    = n_conserved_variables<dim, n_species>;
    const auto    material_ = material<n_species>();

    const auto q = reference_primitive_state<dim, n_species>(dealii::Point<dim>(0.3, -0.2, 0.1));
    const ConservedVariablesType<dim, double, n_species> w =
      NRBC::conserved_variables<dim, n_species, double>(q, material_);

    ConservedVariablesGradientType<dim, double, n_species> grad_w;
    for (unsigned int c = 0; c < n_vars; ++c)
      for (unsigned int d = 0; d < dim; ++d)
        grad_w[c][d] = 0.1 * (c + 1) - 0.07 * (d + 1) * (c % 2 == 0 ? 1. : -1.);

    const NSpeciesDofStateView<dim,
                               n_species,
                               double,
                               const ConservedVariablesType<dim, double, n_species>>
      state(w, material_);

    // primitive values
    const auto q_roundtrip = NRBC::primitive_variables<dim, n_species, double>(state);
    expect_near(q_roundtrip, q);

    // gradients
    const auto dp               = NRBC::pressure_derivatives<dim, n_species, double>(state);
    const auto grad_q           = NRBC::primitive_gradient<dim, n_species, double>(w, grad_w, dp);
    const auto grad_w_roundtrip = NRBC::conserved_gradient<dim, n_species, double>(w, grad_q, dp);
    expect_near(grad_w_roundtrip, grad_w);

    // gradient of the velocity by the quotient rule
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int d = 0; d < dim; ++d)
        expect_near(grad_q[NRBC::PrimitiveIndex<dim>::velocity + i][d],
                    (grad_w[1 + i][d] - w[1 + i] / w[0] * grad_w[0][d]) / w[0]);
  }

  TEST(NonReflectingBoundaryConditions, PrimitiveConservedRoundTripSingleSpecies)
  {
    check_primitive_conserved_round_trip<1>();
  }

  TEST(NonReflectingBoundaryConditions, PrimitiveConservedRoundTripTwoSpecies)
  {
    check_primitive_conserved_round_trip<2>();
  }

  TEST(NonReflectingBoundaryConditions, PressureDerivativesTwoSpecies)
  {
    constexpr int dim       = 2;
    constexpr int n_species = 2;
    using State             = ConservedVariablesType<dim, double, n_species>;
    const auto material_    = two_species_material();

    const auto  q = reference_primitive_state<dim, n_species>(dealii::Point<dim>(0.3, -0.2));
    const State w = NRBC::conserved_variables<dim, n_species, double>(q, material_);

    const auto dp = NRBC::pressure_derivatives<dim, n_species, double>(
      NSpeciesDofStateView<dim, n_species, double, const State>(w, material_));

    // central finite differences of the pressure of the equation of state of the solver
    const double h = 1e-6;
    for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
      {
        State w_plus  = w;
        State w_minus = w;
        w_plus[c] += h;
        w_minus[c] -= h;
        const VectorizedArrayType fd =
          (NSpeciesDofStateView<dim, n_species, double, const State>(w_plus, material_).pressure() -
           NSpeciesDofStateView<dim, n_species, double, const State>(w_minus, material_)
             .pressure()) /
          (2. * h);
        for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
          EXPECT_NEAR(dp[c][v], fd[v], 1e-7) << "component " << c;
      }
  }

  // -----------------------------------------------------------------------------------------------
  // characteristic decomposition
  // -----------------------------------------------------------------------------------------------

  /**
   * Without relaxation (sigma = beta = 0), the normal derivative correction removes exactly the
   * incoming waves and keeps the outgoing waves.
   */
  template <int n_species>
  void
  check_perfectly_non_reflecting(const BoundaryConditionType type)
  {
    constexpr int dim       = 2;
    const auto    material_ = material<n_species>();

    const dealii::Point<dim> normal_(0.6, 0.8);
    const dealii::Point<dim> tangent_(-0.8, 0.6);
    const auto               normal  = to_vectorized(normal_);
    const auto               tangent = to_vectorized(tangent_);

    // outflow: u_n > 0, inflow: u_n < 0
    const double sign = type == BoundaryConditionType::non_reflecting_outflow ? 1. : -1.;
    const auto q = reference_primitive_state<dim, n_species>(sign * 0.3 * normal_ + 0.1 * tangent_);
    const VectorizedArrayType a = speed_of_sound<dim, n_species>(q, material_);

    NRBC::BoundaryParameters<dim, double, n_species> parameters;
    parameters.type             = type;
    parameters.sigma            = 0.;
    parameters.beta             = 0.;
    parameters.target           = q;
    parameters.mean_mach_number = 0.3;

    const VectorizedArrayType c = 10.;

    const CharacteristicWaves<dim, n_species> waves(q, a, normal, tangent);
    const auto correction = [&](const NRBC::PrimitiveVariables<dim, double, n_species> &q_n) {
      return NRBC::normal_derivative_correction<dim, double, n_species>(
        q, normal_gradient<dim, n_species>(q_n, normal), normal, a, c, parameters);
    };

    const NRBC::PrimitiveVariables<dim, double, n_species> zero;
    const bool is_outflow = type == BoundaryConditionType::non_reflecting_outflow;

    {
      SCOPED_TRACE("acoustic wave u_n - a (incoming)");
      expect_near(correction(waves.acoustic_minus), -waves.acoustic_minus);
    }
    {
      SCOPED_TRACE("entropy wave");
      expect_near(correction(waves.entropy), is_outflow ? zero : -waves.entropy);
    }
    {
      SCOPED_TRACE("vorticity wave");
      expect_near(correction(waves.vorticity), is_outflow ? zero : -waves.vorticity);
    }
    if constexpr (n_species > 1)
      {
        SCOPED_TRACE("species wave");
        expect_near(correction(waves.species), is_outflow ? zero : -waves.species);
      }
    {
      SCOPED_TRACE("acoustic wave u_n + a (outgoing)");
      expect_near(correction(waves.acoustic_plus), zero);
    }
    {
      SCOPED_TRACE("superposition");
      NRBC::PrimitiveVariables<dim, double, n_species> expected = -waves.acoustic_minus;
      if (not is_outflow)
        expected -= 2. * waves.entropy + 3. * waves.vorticity + 4. * waves.species;
      expect_near(correction(waves.acoustic_minus + 2. * waves.entropy + 3. * waves.vorticity +
                             4. * waves.species - 5. * waves.acoustic_plus),
                  expected,
                  10.);
    }
  }

  TEST(NonReflectingBoundaryConditions, OutflowPerfectlyNonReflectingSingleSpecies)
  {
    check_perfectly_non_reflecting<1>(BoundaryConditionType::non_reflecting_outflow);
  }

  TEST(NonReflectingBoundaryConditions, OutflowPerfectlyNonReflectingTwoSpecies)
  {
    check_perfectly_non_reflecting<2>(BoundaryConditionType::non_reflecting_outflow);
  }

  TEST(NonReflectingBoundaryConditions, InflowPerfectlyNonReflectingSingleSpecies)
  {
    check_perfectly_non_reflecting<1>(BoundaryConditionType::non_reflecting_inflow);
  }

  TEST(NonReflectingBoundaryConditions, InflowPerfectlyNonReflectingTwoSpecies)
  {
    check_perfectly_non_reflecting<2>(BoundaryConditionType::non_reflecting_inflow);
  }

  TEST(NonReflectingBoundaryConditions, SupersonicOutflowKeepsAllWaves)
  {
    constexpr int dim       = 2;
    constexpr int n_species = 2;
    const auto    material_ = two_species_material();

    const dealii::Point<dim> normal_(1., 0.);
    const auto               normal = to_vectorized(normal_);
    auto q = reference_primitive_state<dim, n_species>(dealii::Point<dim>(0., 0.));
    const VectorizedArrayType a            = speed_of_sound<dim, n_species>(q, material_);
    q[NRBC::PrimitiveIndex<dim>::velocity] = 1.5 * a;

    NRBC::BoundaryParameters<dim, double, n_species> parameters;
    parameters.type   = BoundaryConditionType::non_reflecting_outflow;
    parameters.sigma  = 0.28;
    parameters.beta   = 0.5;
    parameters.target = q;
    parameters.target[NRBC::PrimitiveIndex<dim>::pressure] = 0.5;

    const CharacteristicWaves<dim, n_species> waves(q,
                                                    a,
                                                    normal,
                                                    to_vectorized(dealii::Point<dim>(0., 1.)));
    const auto correction = NRBC::normal_derivative_correction<dim, double, n_species>(
      q,
      normal_gradient<dim, n_species>(waves.acoustic_minus, normal),
      normal,
      a,
      VectorizedArrayType(10.),
      parameters);
    expect_near(correction, NRBC::PrimitiveVariables<dim, double, n_species>());
  }

  // -----------------------------------------------------------------------------------------------
  // relaxation towards the target state
  // -----------------------------------------------------------------------------------------------

  TEST(NonReflectingBoundaryConditions, OutflowPressureRelaxation)
  {
    constexpr int dim       = 2;
    constexpr int n_species = 2;
    using Idx               = NRBC::PrimitiveIndex<dim>;
    const auto material_    = two_species_material();

    const dealii::Point<dim> normal_(-1., 0.);
    const auto               normal = to_vectorized(normal_);
    const auto q = reference_primitive_state<dim, n_species>(dealii::Point<dim>(-0.2, 0.05));
    const VectorizedArrayType a = speed_of_sound<dim, n_species>(q, material_);

    NRBC::BoundaryParameters<dim, double, n_species> parameters;
    parameters.type                  = BoundaryConditionType::non_reflecting_outflow;
    parameters.sigma                 = 0.28;
    parameters.beta                  = 0.;
    parameters.reference_length      = 2.;
    parameters.mean_mach_number      = 0.25;
    parameters.target[Idx::pressure] = 0.9;

    const auto correction = NRBC::normal_derivative_correction<dim, double, n_species>(
      q,
      NRBC::PrimitiveGradient<dim, double, n_species>(),
      normal,
      a,
      VectorizedArrayType(100.),
      parameters);

    // L_1 = sigma (1 - M^2) (p - p_inf) / (2 rho l), dW_1/dn = L_1 / (u_n - a)
    const VectorizedArrayType u_n = 0.2;
    const VectorizedArrayType w_1 =
      0.28 * (1. - 0.25 * 0.25) * (1. - 0.9) / (2. * 1.2 * 2.) / (u_n - a);
    expect_near(correction[Idx::pressure], 1.2 * a * w_1);
    expect_near(correction[Idx::density], 1.2 / a * w_1);
    expect_near(correction[Idx::velocity], -w_1 * normal[0]);
    expect_near(correction[Idx::velocity + 1], VectorizedArrayType(0.));
    expect_near(correction[Idx::mass_fraction], VectorizedArrayType(0.));

    // the pressure on the boundary is decreased towards the target pressure
    for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
      EXPECT_LT(correction[Idx::pressure][v], 0.);
  }

  TEST(NonReflectingBoundaryConditions, InflowRelaxation)
  {
    constexpr int dim       = 2;
    constexpr int n_species = 2;
    using Idx               = NRBC::PrimitiveIndex<dim>;
    const auto material_    = two_species_material();

    const dealii::Point<dim> normal_(0., 1.);
    const auto               normal = to_vectorized(normal_);
    const auto q = reference_primitive_state<dim, n_species>(dealii::Point<dim>(0.1, -0.3));
    const VectorizedArrayType a = speed_of_sound<dim, n_species>(q, material_);

    NRBC::BoundaryParameters<dim, double, n_species> parameters;
    parameters.type                       = BoundaryConditionType::non_reflecting_inflow;
    parameters.sigma                      = 0.5;
    parameters.beta                       = 0.;
    parameters.reference_length           = 1.;
    parameters.mean_mach_number           = 0.;
    parameters.target                     = q;
    parameters.target[Idx::density]       = 1.;
    parameters.target[Idx::velocity]      = 0.;
    parameters.target[Idx::velocity + 1]  = -0.5;
    parameters.target[Idx::mass_fraction] = 0.5;

    const auto correction = NRBC::normal_derivative_correction<dim, double, n_species>(
      q,
      NRBC::PrimitiveGradient<dim, double, n_species>(),
      normal,
      a,
      VectorizedArrayType(100.),
      parameters);

    const VectorizedArrayType u_n = -0.3;
    // acoustic wave relaxes the normal velocity
    const VectorizedArrayType w_1 = 0.5 * (-0.5 * a * (u_n - (-0.5))) / (u_n - a);
    // entropy, vorticity and species waves
    const VectorizedArrayType w_2 = 0.5 * a * (1.2 - 1.) / u_n;
    const VectorizedArrayType w_3 = 0.5 * a * (0.1 - 0.) / u_n;
    const VectorizedArrayType w_y = 0.5 * a * (0.3 - 0.5) / u_n;

    expect_near(correction[Idx::density], 1.2 / a * w_1 + w_2);
    expect_near(correction[Idx::velocity], w_3);
    expect_near(correction[Idx::velocity + 1], -w_1);
    expect_near(correction[Idx::pressure], 1.2 * a * w_1);
    expect_near(correction[Idx::mass_fraction], w_y);

    // all quantities are corrected towards the target state
    for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
      {
        EXPECT_LT(correction[Idx::velocity + 1][v], 0.);  // u_n = -0.3 > -0.5
        EXPECT_LT(w_2[v], 0.);                            // rho = 1.2 > 1
        EXPECT_LT(correction[Idx::velocity][v], 0.);      // u_t = 0.1 > 0
        EXPECT_GT(correction[Idx::mass_fraction][v], 0.); // Y = 0.3 < 0.5
      }
  }

  TEST(NonReflectingBoundaryConditions, IncomingEigenvalueSafeguard)
  {
    constexpr int dim       = 2;
    constexpr int n_species = 1;
    using Idx               = NRBC::PrimitiveIndex<dim>;
    const auto material_    = single_species_material();

    const auto normal = to_vectorized(dealii::Point<dim>(1., 0.));
    // vanishing normal velocity at the inflow
    const auto q = reference_primitive_state<dim, n_species>(dealii::Point<dim>(0., 0.));
    const VectorizedArrayType a = speed_of_sound<dim, n_species>(q, material_);

    NRBC::BoundaryParameters<dim, double, n_species> parameters;
    parameters.type                 = BoundaryConditionType::non_reflecting_inflow;
    parameters.sigma                = 0.3;
    parameters.beta                 = 0.;
    parameters.target               = q;
    parameters.target[Idx::density] = 1.;

    const VectorizedArrayType c = 20.;
    const auto correction       = NRBC::normal_derivative_correction<dim, double, n_species>(
      q, NRBC::PrimitiveGradient<dim, double, n_species>(), normal, a, c, parameters);

    // The eigenvalue is limited to sigma a / (l c), such that the boundary value correction
    // (rho - rho_inf) / c / (sigma a / (l c)) * sigma a / l is exactly the deviation from the
    // target.
    expect_near(correction[Idx::density] / c, VectorizedArrayType(-(1.2 - 1.)));
  }

  TEST(NonReflectingBoundaryConditions, TransverseTerms)
  {
    constexpr int dim       = 2;
    constexpr int n_species = 2;
    using Idx               = NRBC::PrimitiveIndex<dim>;
    const auto material_    = two_species_material();

    const auto normal = to_vectorized(dealii::Point<dim>(1., 0.));
    const auto q      = reference_primitive_state<dim, n_species>(dealii::Point<dim>(0.2, 0.3));
    const VectorizedArrayType a   = speed_of_sound<dim, n_species>(q, material_);
    const VectorizedArrayType rho = q[Idx::density];
    const VectorizedArrayType v   = q[Idx::velocity + 1];

    // only tangential derivatives
    NRBC::PrimitiveGradient<dim, double, n_species> gradient;
    const VectorizedArrayType                       rho_y = 0.3, u_y = -0.4, v_y = 0.7, p_y = 1.1;
    gradient[Idx::density][1]       = rho_y;
    gradient[Idx::velocity][1]      = u_y;
    gradient[Idx::velocity + 1][1]  = v_y;
    gradient[Idx::pressure][1]      = p_y;
    gradient[Idx::mass_fraction][1] = 0.9;

    NRBC::BoundaryParameters<dim, double, n_species> parameters;
    parameters.type                  = BoundaryConditionType::non_reflecting_outflow;
    parameters.sigma                 = 0.;
    parameters.beta                  = 0.75;
    parameters.target[Idx::pressure] = 1.;

    const auto correction = NRBC::normal_derivative_correction<dim, double, n_species>(
      q, gradient, normal, a, VectorizedArrayType(100.), parameters);

    // Eq. (15) of the reference
    const VectorizedArrayType T_1 = v * (p_y / (2. * rho * a) - 0.5 * u_y) + 0.5 * a * v_y;
    const VectorizedArrayType w_1 = -0.75 * T_1 / (q[Idx::velocity] - a);
    expect_near(correction[Idx::pressure], rho * a * w_1);
    expect_near(correction[Idx::velocity], -w_1);
    expect_near(correction[Idx::velocity + 1], VectorizedArrayType(0.));
    expect_near(correction[Idx::density], rho / a * w_1);
    expect_near(correction[Idx::mass_fraction], VectorizedArrayType(0.));
  }

  // -----------------------------------------------------------------------------------------------
  // least-squares projection on the faces
  // -----------------------------------------------------------------------------------------------

  TEST(NonReflectingBoundaryConditions, LeastSquaresProjection1D)
  {
    const unsigned int degree   = 3;
    const unsigned int n_points = 6;
    const auto         points   = dealii::QGauss<1>(n_points).get_points();
    const auto         P        = NRBC::least_squares_projection_matrix<double>(points, degree);

    for (unsigned int i = 0; i < n_points; ++i)
      for (unsigned int j = 0; j < n_points; ++j)
        {
          // symmetric
          EXPECT_NEAR(P[i * n_points + j], P[j * n_points + i], tolerance);
          // idempotent
          double PP = 0.;
          for (unsigned int k = 0; k < n_points; ++k)
            PP += P[i * n_points + k] * P[k * n_points + j];
          EXPECT_NEAR(PP, P[i * n_points + j], tolerance);
        }

    // polynomials up to the degree are reproduced, higher degrees are not
    for (unsigned int k = 0; k <= degree + 1; ++k)
      {
        double max_error = 0.;
        for (unsigned int i = 0; i < n_points; ++i)
          {
            double projected = 0.;
            for (unsigned int j = 0; j < n_points; ++j)
              projected += P[i * n_points + j] * std::pow(points[j][0], k);
            max_error = std::max(max_error, std::abs(projected - std::pow(points[i][0], k)));
          }
        if (k <= degree)
          EXPECT_LT(max_error, tolerance) << "degree " << k;
        else
          EXPECT_GT(max_error, 1e-4) << "degree " << k;
      }
  }

  TEST(NonReflectingBoundaryConditions, FaceProjection3D)
  {
    constexpr int      dim      = 3;
    const unsigned int degree   = 2;
    const unsigned int n_points = 4;
    const auto         points   = dealii::QGauss<1>(n_points).get_points();
    const auto         P        = NRBC::least_squares_projection_matrix<double>(points, degree);

    const auto f = [](const double x, const double y) { return 1. + x * x * y - 2. * y * y; };
    const auto g = [](const double x, const double y) { return x * x * x * y; };

    dealii::AlignedVector<dealii::Tensor<1, 2, VectorizedArrayType>> values(n_points * n_points);
    for (unsigned int i1 = 0; i1 < n_points; ++i1)
      for (unsigned int i0 = 0; i0 < n_points; ++i0)
        {
          values[i0 + n_points * i1][0] = f(points[i0][0], points[i1][0]);
          values[i0 + n_points * i1][1] = g(points[i0][0], points[i1][0]);
        }
    const auto original = values;

    NRBC::apply_face_projection<dim>(P, n_points, values);

    double max_deviation_g = 0.;
    for (unsigned int q = 0; q < values.size(); ++q)
      {
        // f is in the tensor-product space of degree 2 and therefore reproduced
        expect_near(values[q][0], original[q][0]);
        max_deviation_g = std::max(max_deviation_g, std::abs(values[q][1][0] - original[q][1][0]));
      }
    // g is cubic in x and therefore changed
    EXPECT_GT(max_deviation_g, 1e-4);

    // the projection is idempotent
    const auto projected = values;
    NRBC::apply_face_projection<dim>(P, n_points, values);
    for (unsigned int q = 0; q < values.size(); ++q)
      expect_near(values[q], projected[q]);
  }
} // namespace
