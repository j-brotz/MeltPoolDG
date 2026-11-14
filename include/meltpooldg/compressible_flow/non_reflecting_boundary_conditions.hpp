#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/non_reflecting_boundary_data.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/state_views_n_species.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

/**
 * @file
 *
 * Polynomial-correction Navier-Stokes characteristic boundary conditions (PC-NSCBC) for the
 * multi-species compressible flow solver following
 *
 *   E.K. Shehadi and E.T.A. van der Weide, A polynomial-correction Navier-Stokes characteristic
 *   boundary condition, Comput. Phys. Commun. 303 (2024) 109194.
 *
 * The method consists of three steps on each boundary face:
 *
 * 1. Characteristic decomposition: The normal derivative of the primitive variables
 *    \f$\boldsymbol{Q} = (\rho, \boldsymbol{u}, p, Y_1, \dots, Y_{n-1})\f$ of the interior
 * solution, i.e. of the Lagrange interpolant of the primitive variables in the nodes, is
 * transformed into the normal derivatives \f$\partial_n \mathcal{W} = \boldsymbol{S}^{-1}
 * \partial_n \boldsymbol{Q}\f$ of the characteristic variables, i.e. the acoustic waves with the
 * eigenvalues \f$u_n \mp a\f$, the entropy wave, the vorticity waves and the species waves, which
 * all travel with \f$u_n\f$. All quantities are expressed in the frame of the outward-pointing
 * normal vector, such that outgoing waves have positive and incoming waves have negative
 * eigenvalues. Since the mass fractions are advected with the flow and the heat capacity ratio of
 * the mixture is a function of the mass fractions only, the pressure evolves according to \f$Dp/Dt
 * = -\rho a^2 \nabla \cdot \boldsymbol{u}\f$ also for multiple species, and the species waves
 * simply extend the system of the single-species case.
 *
 * 2. Modification of the incoming wave amplitudes: The amplitudes of the incoming waves are
 *    replaced by \f$\mathcal{L}^+ = \sigma \mathcal{K} - \beta \mathcal{T}^+\f$, which relaxes the
 *    boundary towards the prescribed target state and partially accounts for the transverse terms.
 *    This gives the boundary-conforming normal derivative \f$\overline{\boldsymbol{Q}}_n =
 *    \boldsymbol{S} \boldsymbol{\Lambda}^{-1} \overline{\mathcal{L}}\f$.
 *
 * 3. Polynomial-correction reconstruction: The nodal values of the solution in the interior of the
 *    element are fixed and the nodal values on the boundary face are chosen such that the normal
 *    derivative of the solution is the boundary-conforming one in the least-squares sense on the
 *    face quadrature points. For tensor-product elements with nodes on the element boundary, the
 *    contribution of the face nodes to the normal derivative is \f$c \, \boldsymbol{Q}^\Gamma\f$,
 *    where \f$c\f$ is the normal derivative of the one-dimensional Lagrange polynomial of the face
 *    nodes on the face. Thus, the boundary state is the interior trace plus the least-squares
 *    projection of \f$(\overline{\boldsymbol{Q}}_n - \boldsymbol{Q}_n) / c\f$ onto the polynomials
 *    on the face, see Eqs. (A.9)-(A.16) of the reference.
 *
 * The boundary state then enters the numerical flux. For viscous flows, the gradient of the
 * boundary state is the gradient of the interior primitive variables with the normal derivative
 * replaced by the boundary-conforming one, see Appendix A.2 of the reference, where the tangential
 * derivative of the correction is neglected.
 *
 * Two types of boundaries are supported:
 * - Outflow with a prescribed static pressure \f$p_\infty\f$. The incoming acoustic wave is
 *   \f$\mathcal{L}_1^+ = \sigma (1 - \overline{M}^2) (p - p_\infty) / (2 \rho l) - \beta
 *   \mathcal{T}_1\f$, see Eqs. (23)-(24).
 * - Inflow with a prescribed static density, velocity and mass fractions. The incoming waves are
 *   the acoustic wave, which relaxes the normal velocity, as well as the entropy, vorticity and
 *   species waves, which relax the density, the tangential velocity and the mass fractions,
 *   respectively, see Eqs. (25)-(26). As in the reference, the transverse terms are neglected at
 *   inflow boundaries. The prescribed energy is only used for supersonic inflow, where all waves
 *   enter the domain and the prescribed state is imposed.
 *
 * At supersonic outflow, all waves leave the domain and the interior state is used.
 *
 * @note The incoming wave amplitudes are converted into normal derivatives by dividing by the
 * corresponding eigenvalues, which vanish for sonic acoustic waves or a vanishing normal velocity
 * at an inflow. As a safeguard, the magnitude of the eigenvalues of incoming waves is limited from
 * below by \f$\max(\sigma a / (l c), 10^{-3} a)\f$. The first bound ensures that the relaxation
 * does not correct the boundary state beyond the target state.
 */
namespace MeltPoolDG::CompressibleFlow::NonReflectingBoundary
{
  /// Compile-time indices of the primitive variables \f$(\rho, \boldsymbol{u}, p, Y_1, \dots,
  /// Y_{n-1})\f$ used for the characteristic analysis. The layout is the same as for the conserved
  /// variables, such that the conserved variable types can be used for storage.
  template <int dim>
  struct PrimitiveIndex
  {
    static constexpr unsigned int density       = 0;
    static constexpr unsigned int velocity      = 1;
    static constexpr unsigned int pressure      = dim + 1;
    static constexpr unsigned int mass_fraction = dim + 2;
  };

  /// Primitive variables \f$(\rho, \boldsymbol{u}, p, Y_1, \dots, Y_{n-1})\f$.
  template <int dim, typename number, int n_species>
  using PrimitiveVariables = ConservedVariablesType<dim, number, n_species>;

  /// Gradient of the primitive variables.
  template <int dim, typename number, int n_species>
  using PrimitiveGradient = ConservedVariablesGradientType<dim, number, n_species>;

  /// Lower bound of the magnitude of the eigenvalues of incoming waves relative to the speed of
  /// sound, which avoids a division by zero.
  constexpr double minimum_relative_incoming_eigenvalue = 1e-3;

  /**
   * @brief Check whether the equation of state is supported by the non-reflecting boundary
   * conditions.
   *
   * The conversion between the primitive and the conserved variables requires the derivatives of
   * the pressure with respect to the conserved variables, which are available for ideal and
   * stiffened gases with a single species and for ideal gas mixtures.
   */
  template <typename number>
  bool
  is_supported_equation_of_state(const MaterialPhaseData<number> &material)
  {
    return material.eos_type == EquationOfState::ideal_gas or
           (material.eos_type == EquationOfState::stiffened_gas and
            material.number_of_species == 1);
  }

  /**
   * @brief Derivatives of the pressure with respect to the conserved variables.
   *
   * For an ideal (or stiffened) gas \f$p = (\gamma - 1) \rho e (- \gamma p_\infty)\f$, where the
   * heat capacity ratio of a mixture is the mass-fraction-weighted average \f$\gamma = \sum_k Y_k
   * \gamma_k\f$ of the species heat capacity ratios, as used by the equation of state of the
   * solver.
   *
   * @param state State view of the conserved variables including the material data.
   *
   * @return Derivatives \f$\partial p / \partial \boldsymbol{U}\f$ in the layout of the conserved
   * variables.
   */
  template <int dim, int n_species, typename number, typename StateView>
  ConservedVariablesType<dim, number, n_species>
  pressure_derivatives(const StateView &state)
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using Idx                 = ConservedVariableIndex<dim>;

    const auto                velocity                 = state.velocity();
    const VectorizedArrayType heat_capacity_ratio      = state.heat_capacity_ratio();
    const VectorizedArrayType heat_capacity_ratio_m1   = heat_capacity_ratio - 1.;
    const VectorizedArrayType velocity_magnitude_sqr_2 = 0.5 * (velocity * velocity);

    ConservedVariablesType<dim, number, n_species> derivatives;
    derivatives[Idx::density] = heat_capacity_ratio_m1 * velocity_magnitude_sqr_2;
    for (unsigned int d = 0; d < dim; ++d)
      derivatives[Idx::momentum + d] = -heat_capacity_ratio_m1 * velocity[d];
    derivatives[Idx::energy] = heat_capacity_ratio_m1;

    if constexpr (n_species > 1)
      {
        // the heat capacity ratio depends on the partial densities and the density
        const VectorizedArrayType specific_inner_energy =
          state.pressure() / (state.density() * heat_capacity_ratio_m1);
        const number heat_capacity_ratio_last = state.heat_capacity_ratio(n_species - 1);

        derivatives[Idx::density] -=
          specific_inner_energy * (heat_capacity_ratio - heat_capacity_ratio_last);
        for (unsigned int species = 0; species < n_species - 1; ++species)
          derivatives[n_conserved_variables<dim> + species] =
            specific_inner_energy * (state.heat_capacity_ratio(species) - heat_capacity_ratio_last);
      }

    return derivatives;
  }

  /**
   * @brief Compute the primitive variables from the conserved ones.
   *
   * @param state State view of the conserved variables including the material data.
   */
  template <int dim, int n_species, typename number, typename StateView>
  PrimitiveVariables<dim, number, n_species>
  primitive_variables(const StateView &state)
  {
    using Idx = PrimitiveIndex<dim>;

    PrimitiveVariables<dim, number, n_species> primitive;
    primitive[Idx::density] = state.density();
    for (unsigned int d = 0; d < dim; ++d)
      primitive[Idx::velocity + d] = state.velocity(d);
    primitive[Idx::pressure] = state.pressure();
    if constexpr (n_species > 1)
      for (unsigned int species = 0; species < n_species - 1; ++species)
        primitive[Idx::mass_fraction + species] = state.mass_fraction(species);

    return primitive;
  }

  /**
   * @brief Compute the gradient of the primitive variables from the gradient of the conserved
   * ones.
   *
   * @param conserved Conserved variables.
   * @param conserved_gradient Gradient of the conserved variables.
   * @param pressure_derivatives Derivatives of the pressure with respect to the conserved
   * variables, see pressure_derivatives().
   */
  template <int dim, int n_species, typename number>
  PrimitiveGradient<dim, number, n_species>
  primitive_gradient(
    const ConservedVariablesType<dim, number, n_species>         &conserved,
    const ConservedVariablesGradientType<dim, number, n_species> &conserved_gradient,
    const ConservedVariablesType<dim, number, n_species>         &pressure_derivatives)
  {
    using Idx  = PrimitiveIndex<dim>;
    using CIdx = ConservedVariableIndex<dim>;

    const dealii::VectorizedArray<number> inverse_density = 1. / conserved[CIdx::density];

    PrimitiveGradient<dim, number, n_species> gradient;
    gradient[Idx::density] = conserved_gradient[CIdx::density];
    for (unsigned int d = 0; d < dim; ++d)
      gradient[Idx::velocity + d] =
        (conserved_gradient[CIdx::momentum + d] -
         (conserved[CIdx::momentum + d] * inverse_density) * conserved_gradient[CIdx::density]) *
        inverse_density;

    gradient[Idx::pressure] = pressure_derivatives[0] * conserved_gradient[0];
    for (unsigned int c = 1; c < n_conserved_variables<dim, n_species>; ++c)
      gradient[Idx::pressure] += pressure_derivatives[c] * conserved_gradient[c];

    if constexpr (n_species > 1)
      for (unsigned int species = 0; species < n_species - 1; ++species)
        {
          const unsigned int c = n_conserved_variables<dim> + species;
          gradient[Idx::mass_fraction + species] =
            (conserved_gradient[c] - (conserved[c] * inverse_density) * conserved_gradient[0]) *
            inverse_density;
        }

    return gradient;
  }

  /**
   * @brief Compute the conserved variables from the primitive ones.
   *
   * @param primitive Primitive variables.
   * @param material Material data defining the equation of state.
   */
  template <int dim, int n_species, typename number>
  ConservedVariablesType<dim, number, n_species>
  conserved_variables(const PrimitiveVariables<dim, number, n_species> &primitive,
                      const MaterialPhaseData<number>                  &material)
  {
    using Idx  = PrimitiveIndex<dim>;
    using CIdx = ConservedVariableIndex<dim>;

    ConservedVariablesType<dim, number, n_species> conserved;
    const dealii::VectorizedArray<number>          density = primitive[Idx::density];
    conserved[CIdx::density]                               = density;

    dealii::VectorizedArray<number> kinetic_energy = 0.;
    for (unsigned int d = 0; d < dim; ++d)
      {
        conserved[CIdx::momentum + d] = density * primitive[Idx::velocity + d];
        kinetic_energy += primitive[Idx::velocity + d] * primitive[Idx::velocity + d];
      }
    kinetic_energy *= 0.5 * density;

    if constexpr (n_species > 1)
      for (unsigned int species = 0; species < n_species - 1; ++species)
        conserved[n_conserved_variables<dim> + species] =
          density * primitive[Idx::mass_fraction + species];

    // The inner energy depends on the density and mass fractions, which are already set.
    const NSpeciesDofStateView<dim,
                               n_species,
                               number,
                               const ConservedVariablesType<dim, number, n_species>>
                                          state(conserved, material);
    const dealii::VectorizedArray<number> inner_energy =
      state.inner_energy_from_pressure(primitive[Idx::pressure]);

    conserved[CIdx::energy] = inner_energy + kinetic_energy;

    return conserved;
  }

  /**
   * @brief Compute the gradient of the conserved variables from the gradient of the primitive
   * ones.
   *
   * @param conserved Conserved variables.
   * @param gradient Gradient of the primitive variables.
   * @param pressure_derivatives Derivatives of the pressure with respect to the conserved
   * variables, see pressure_derivatives().
   */
  template <int dim, int n_species, typename number>
  ConservedVariablesGradientType<dim, number, n_species>
  conserved_gradient(const ConservedVariablesType<dim, number, n_species> &conserved,
                     const PrimitiveGradient<dim, number, n_species>      &gradient,
                     const ConservedVariablesType<dim, number, n_species> &pressure_derivatives)
  {
    using Idx  = PrimitiveIndex<dim>;
    using CIdx = ConservedVariableIndex<dim>;

    const dealii::VectorizedArray<number> density         = conserved[CIdx::density];
    const dealii::VectorizedArray<number> inverse_density = 1. / density;

    ConservedVariablesGradientType<dim, number, n_species> result;
    result[CIdx::density] = gradient[Idx::density];
    for (unsigned int d = 0; d < dim; ++d)
      result[CIdx::momentum + d] =
        (conserved[CIdx::momentum + d] * inverse_density) * gradient[Idx::density] +
        density * gradient[Idx::velocity + d];

    if constexpr (n_species > 1)
      for (unsigned int species = 0; species < n_species - 1; ++species)
        {
          const unsigned int c = n_conserved_variables<dim> + species;
          result[c]            = (conserved[c] * inverse_density) * gradient[Idx::density] +
                      density * gradient[Idx::mass_fraction + species];
        }

    // invert the pressure gradient for the gradient of the total energy
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> energy_gradient =
      gradient[Idx::pressure];
    for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
      if (c != CIdx::energy)
        energy_gradient -= pressure_derivatives[c] * result[c];
    result[CIdx::energy] = energy_gradient / pressure_derivatives[CIdx::energy];

    return result;
  }

  /**
   * @brief Target state and tuning parameters of a non-reflecting boundary at a quadrature point.
   */
  template <int dim, typename number, int n_species>
  struct BoundaryParameters
  {
    /// Type of the boundary, either BoundaryConditionType::non_reflecting_inflow or
    /// BoundaryConditionType::non_reflecting_outflow.
    BoundaryConditionType type = BoundaryConditionType::non_reflecting_outflow;

    /// Prescribed primitive state. At outflow boundaries, only the pressure is used. At inflow
    /// boundaries, the density, the velocity and the mass fractions are used.
    PrimitiveVariables<dim, number, n_species> target;

    /// Normal relaxation coefficient \f$\sigma\f$.
    number sigma = 0.;

    /// Transverse relaxation coefficient \f$\beta\f$.
    number beta = 0.;

    /// Reference length of the relaxation.
    number reference_length = 1.;

    /// Mean Mach number on the boundary.
    number mean_mach_number = 0.;
  };

  /**
   * @brief Compute the correction of the normal derivative of the primitive variables, which turns
   * the normal derivative of the interior solution into the boundary-conforming one.
   *
   * @param primitive Primitive variables of the interior solution.
   * @param gradient Gradient of the primitive variables of the interior solution.
   * @param normal Outward-pointing unit normal vector.
   * @param speed_of_sound Speed of sound of the interior solution.
   * @param boundary_basis_normal_derivative Normal derivative \f$c\f$ of the Lagrange polynomial of
   * the face nodes on the face, which is only used for the safeguard of the eigenvalues.
   * @param parameters Target state and tuning parameters of the boundary.
   *
   * @return The correction \f$\overline{\boldsymbol{Q}}_n - \boldsymbol{Q}_n =
   * \boldsymbol{S} (\boldsymbol{\Lambda}^{-1} \overline{\mathcal{L}} - \boldsymbol{\Lambda}^{-1}
   * \mathcal{L})\f$, which is nonzero only for the components of the incoming waves.
   */
  template <int dim, typename number, int n_species>
  PrimitiveVariables<dim, number, n_species>
  normal_derivative_correction(
    const PrimitiveVariables<dim, number, n_species>              &primitive,
    const PrimitiveGradient<dim, number, n_species>               &gradient,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    const dealii::VectorizedArray<number>                         &speed_of_sound,
    const dealii::VectorizedArray<number>                         &boundary_basis_normal_derivative,
    const BoundaryParameters<dim, number, n_species>              &parameters)
  {
    using VectorizedArrayType               = dealii::VectorizedArray<number>;
    using VectorType                        = dealii::Tensor<1, dim, VectorizedArrayType>;
    using Idx                               = PrimitiveIndex<dim>;
    constexpr unsigned int n_mass_fractions = n_species - 1;

    Assert(parameters.type == BoundaryConditionType::non_reflecting_inflow or
             parameters.type == BoundaryConditionType::non_reflecting_outflow,
           dealii::ExcMessage("The boundary type is not a non-reflecting boundary."));

    const VectorizedArrayType density = primitive[Idx::density];
    const VectorizedArrayType a       = speed_of_sound;
    VectorType                velocity;
    for (unsigned int d = 0; d < dim; ++d)
      velocity[d] = primitive[Idx::velocity + d];
    const VectorizedArrayType normal_velocity = velocity * normal;

    // normal derivatives of the primitive variables
    VectorType velocity_n;
    for (unsigned int d = 0; d < dim; ++d)
      velocity_n[d] = gradient[Idx::velocity + d] * normal;
    const VectorizedArrayType normal_velocity_n = velocity_n * normal;
    const VectorizedArrayType density_n         = gradient[Idx::density] * normal;
    const VectorizedArrayType pressure_n        = gradient[Idx::pressure] * normal;

    const VectorizedArrayType inverse_2_rho_a = 1. / (2. * density * a);
    const VectorizedArrayType inverse_a_sqr   = 1. / (a * a);

    // normal derivatives of the characteristic variables W = S^{-1} Q_n
    const VectorizedArrayType w_acoustic  = pressure_n * inverse_2_rho_a - 0.5 * normal_velocity_n;
    const VectorizedArrayType w_entropy   = density_n - pressure_n * inverse_a_sqr;
    const VectorType          w_vorticity = velocity_n - normal_velocity_n * normal;
    std::array<VectorizedArrayType, n_mass_fractions> w_species;
    for (unsigned int k = 0; k < n_mass_fractions; ++k)
      w_species[k] = gradient[Idx::mass_fraction + k] * normal;

    // transverse terms T = S^{-1} B dQ/dt of the incoming waves
    const number                                      beta                = parameters.beta;
    VectorizedArrayType                               transverse_acoustic = 0.;
    VectorizedArrayType                               transverse_entropy  = 0.;
    VectorType                                        transverse_vorticity;
    std::array<VectorizedArrayType, n_mass_fractions> transverse_species;
    transverse_species.fill(VectorizedArrayType(0.));
    if (beta != 0.)
      {
        const VectorType tangential_velocity = velocity - normal_velocity * normal;

        // tangential divergence of the velocity
        VectorizedArrayType divergence = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          divergence += gradient[Idx::velocity + d][d];
        const VectorizedArrayType tangential_divergence = divergence - normal_velocity_n;

        // tangential convection of the velocity (u_t . grad) u
        VectorType convection;
        for (unsigned int d = 0; d < dim; ++d)
          convection[d] = tangential_velocity * gradient[Idx::velocity + d];

        const VectorizedArrayType transverse_normal_velocity = convection * normal;
        const VectorType transverse_velocity = convection + gradient[Idx::pressure] / density;
        const VectorizedArrayType transverse_pressure =
          tangential_velocity * gradient[Idx::pressure] + density * a * a * tangential_divergence;
        const VectorizedArrayType transverse_density =
          tangential_velocity * gradient[Idx::density] + density * tangential_divergence;

        transverse_acoustic =
          transverse_pressure * inverse_2_rho_a - 0.5 * transverse_normal_velocity;
        transverse_entropy   = transverse_density - transverse_pressure * inverse_a_sqr;
        transverse_vorticity = transverse_velocity - (transverse_velocity * normal) * normal;
        for (unsigned int k = 0; k < n_mass_fractions; ++k)
          transverse_species[k] = tangential_velocity * gradient[Idx::mass_fraction + k];
      }

    // eigenvalues of incoming waves with a safeguard against a division by zero
    const number              sigma  = parameters.sigma;
    const number              length = parameters.reference_length;
    const VectorizedArrayType minimum_eigenvalue =
      std::max(sigma * a / (length * boundary_basis_normal_derivative),
               VectorizedArrayType(minimum_relative_incoming_eigenvalue) * a);
    const auto incoming_eigenvalue = [&](const VectorizedArrayType &eigenvalue) {
      return std::min(eigenvalue, -minimum_eigenvalue);
    };

    const number one_minus_mach_sqr =
      std::max(1. - parameters.mean_mach_number * parameters.mean_mach_number, 0.);

    // The acoustic wave with the eigenvalue u_n - a is incoming for u_n < a.
    const VectorizedArrayType acoustic_eigenvalue = incoming_eigenvalue(normal_velocity - a);
    const auto                mask_subsonic       = [&](const VectorizedArrayType &value) {
      return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
        normal_velocity, a, value, VectorizedArrayType(0.));
    };

    // corrections of the normal derivatives of the characteristic variables
    VectorizedArrayType                               delta_acoustic = 0.;
    VectorizedArrayType                               delta_entropy  = 0.;
    VectorType                                        delta_vorticity;
    std::array<VectorizedArrayType, n_mass_fractions> delta_species;
    delta_species.fill(VectorizedArrayType(0.));

    if (parameters.type == BoundaryConditionType::non_reflecting_outflow)
      {
        // static pressure outlet, Eqs. (23)-(24)
        const VectorizedArrayType penalty =
          one_minus_mach_sqr * (primitive[Idx::pressure] - parameters.target[Idx::pressure]) /
          (2. * density * length);
        const VectorizedArrayType amplitude = sigma * penalty - beta * transverse_acoustic;
        delta_acoustic = mask_subsonic(amplitude / acoustic_eigenvalue - w_acoustic);
      }
    else
      {
        // static inlet, Eqs. (25)-(26), expressed in the frame of the outward normal
        VectorType target_velocity;
        for (unsigned int d = 0; d < dim; ++d)
          target_velocity[d] = parameters.target[Idx::velocity + d];
        const VectorizedArrayType target_normal_velocity = target_velocity * normal;
        const VectorType          target_tangential_velocity =
          target_velocity - target_normal_velocity * normal;

        const VectorizedArrayType penalty_acoustic =
          -0.5 * a / length * one_minus_mach_sqr * (normal_velocity - target_normal_velocity);
        const VectorizedArrayType amplitude_acoustic =
          sigma * penalty_acoustic - beta * transverse_acoustic;
        delta_acoustic = mask_subsonic(amplitude_acoustic / acoustic_eigenvalue - w_acoustic);

        // the entropy, vorticity and species waves travel with u_n < 0
        const VectorizedArrayType convective_eigenvalue = incoming_eigenvalue(normal_velocity);
        const VectorizedArrayType inverse_eigenvalue    = 1. / convective_eigenvalue;
        const VectorizedArrayType relaxation_rate       = sigma * a / length;

        const VectorizedArrayType amplitude_entropy =
          relaxation_rate * (density - parameters.target[Idx::density]) - beta * transverse_entropy;
        delta_entropy = amplitude_entropy * inverse_eigenvalue - w_entropy;

        const VectorType amplitude_vorticity =
          relaxation_rate * ((velocity - normal_velocity * normal) - target_tangential_velocity) -
          beta * transverse_vorticity;
        delta_vorticity = amplitude_vorticity * inverse_eigenvalue - w_vorticity;

        for (unsigned int k = 0; k < n_mass_fractions; ++k)
          {
            const VectorizedArrayType amplitude_species =
              relaxation_rate *
                (primitive[Idx::mass_fraction + k] - parameters.target[Idx::mass_fraction + k]) -
              beta * transverse_species[k];
            delta_species[k] = amplitude_species * inverse_eigenvalue - w_species[k];
          }
      }

    // transform back to the primitive variables: Q_n = S W_n
    PrimitiveVariables<dim, number, n_species> correction;
    correction[Idx::density] = density / a * delta_acoustic + delta_entropy;
    for (unsigned int d = 0; d < dim; ++d)
      correction[Idx::velocity + d] = -delta_acoustic * normal[d] + delta_vorticity[d];
    correction[Idx::pressure] = density * a * delta_acoustic;
    for (unsigned int k = 0; k < n_mass_fractions; ++k)
      correction[Idx::mass_fraction + k] = delta_species[k];

    return correction;
  }

  /**
   * @brief Least-squares projection onto the polynomials of a given degree in the quadrature points
   * of a one-dimensional quadrature rule.
   *
   * @param points The quadrature points in the unit interval.
   * @param degree The polynomial degree.
   *
   * @return The projection matrix \f$\boldsymbol{V} (\boldsymbol{V}^T \boldsymbol{V})^{-1}
   * \boldsymbol{V}^T\f$ in row-major format, where \f$\boldsymbol{V}\f$ contains the values of a
   * polynomial basis in the quadrature points. Since the column space of \f$\boldsymbol{V}\f$ is
   * independent of the basis, the projection equals the one of Eq. (A.16) with the Lagrange
   * interpolation matrix \f$\boldsymbol{I}\f$, but the Legendre polynomials are better
   * conditioned.
   */
  template <typename number>
  std::vector<number>
  least_squares_projection_matrix(const std::vector<dealii::Point<1>> &points,
                                  const unsigned int                   degree)
  {
    const unsigned int n_points = points.size();
    const unsigned int n_basis  = degree + 1;
    AssertThrow(n_points >= n_basis,
                dealii::ExcMessage("The least-squares projection requires at least as many "
                                   "quadrature points as polynomial basis functions."));

    const auto legendre = dealii::Polynomials::Legendre::generate_complete_basis(degree);

    dealii::FullMatrix<double> values(n_points, n_basis);
    for (unsigned int q = 0; q < n_points; ++q)
      for (unsigned int i = 0; i < n_basis; ++i)
        values(q, i) = legendre[i].value(points[q][0]);

    dealii::FullMatrix<double> gram(n_basis, n_basis);
    values.Tmmult(gram, values);
    gram.gauss_jordan();

    dealii::FullMatrix<double> tmp(n_points, n_basis);
    values.mmult(tmp, gram);
    dealii::FullMatrix<double> projection(n_points, n_points);
    tmp.mTmult(projection, values);

    std::vector<number> result(n_points * n_points);
    for (unsigned int i = 0; i < n_points; ++i)
      for (unsigned int j = 0; j < n_points; ++j)
        result[i * n_points + j] = projection(i, j);
    return result;
  }

  /**
   * @brief Apply the tensor-product least-squares projection to values given in the quadrature
   * points of a face.
   *
   * @param projection_1d One-dimensional projection matrix in row-major format.
   * @param n_points_1d Number of quadrature points in one dimension.
   * @param values Values in the face quadrature points, which are numbered lexicographically. On
   * output, the projected values.
   *
   * @note The projection is the same in both face directions and invariant under reversing the
   * order of the points, since the quadrature points are symmetric. Thus, the orientation of the
   * face coordinate system is irrelevant.
   */
  template <int dim, typename number, typename ValueType>
  void
  apply_face_projection(const std::vector<number>        &projection_1d,
                        const unsigned int                n_points_1d,
                        dealii::AlignedVector<ValueType> &values)
  {
    if constexpr (dim == 1)
      return;
    else
      {
        AssertDimension(projection_1d.size(), n_points_1d * n_points_1d);
        AssertDimension(values.size(), dealii::Utilities::pow(n_points_1d, dim - 1));

        dealii::AlignedVector<ValueType> tmp(values.size());
        for (unsigned int direction = 0; direction < dim - 1; ++direction)
          {
            const unsigned int stride = dealii::Utilities::pow(n_points_1d, direction);
            for (unsigned int q = 0; q < values.size(); ++q)
              {
                // index of the point in the current direction and of the first point of the line
                const unsigned int i     = (q / stride) % n_points_1d;
                const unsigned int start = q - i * stride;
                ValueType          sum   = projection_1d[i * n_points_1d] * values[start];
                for (unsigned int j = 1; j < n_points_1d; ++j)
                  sum += projection_1d[i * n_points_1d + j] * values[start + j * stride];
                tmp[q] = sum;
              }
            values.swap(tmp);
          }
      }
  }
} // namespace MeltPoolDG::CompressibleFlow::NonReflectingBoundary

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Evaluation of the polynomial-correction Navier-Stokes characteristic boundary
   * conditions at the boundary faces of the matrix-free DG discretization.
   *
   * See the documentation of the file non_reflecting_boundary_conditions.hpp for the description of
   * the method.
   *
   * @tparam n_species The number of species.
   */
  template <int dim, typename number, int n_species>
  class NonReflectingBoundaryEvaluator
  {
  public:
    using VectorizedArrayType        = dealii::VectorizedArray<number>;
    using VectorType                 = dealii::LinearAlgebra::distributed::Vector<number>;
    using ConservedVariables         = ConservedVariablesType<dim, number, n_species>;
    using ConservedVariablesGradient = ConservedVariablesGradientType<dim, number, n_species>;
    using PrimitiveVariables = NonReflectingBoundary::PrimitiveVariables<dim, number, n_species>;
    using PrimitiveGradient  = NonReflectingBoundary::PrimitiveGradient<dim, number, n_species>;

    /**
     * @brief Scratch data for the evaluation on the faces.
     *
     * Each thread working on boundary faces needs its own object.
     */
    struct FaceScratch
    {
      /**
       * @brief Constructor.
       *
       * @param flow_scratch_data Reference to the flow scratch data object.
       */
      explicit FaceScratch(const OperationScratchData<dim, number> &flow_scratch_data)
        : phi_basis(flow_scratch_data.scratch_data.get_matrix_free(),
                    true,
                    flow_scratch_data.dof_idx,
                    flow_scratch_data.quad_idx)
        , phi_primitive(flow_scratch_data.scratch_data.get_matrix_free(),
                        true,
                        flow_scratch_data.dof_idx,
                        flow_scratch_data.quad_idx)
      {}

      /// Scalar face evaluator used to compute the normal derivative of the Lagrange polynomial of
      /// the face nodes.
      FEFaceIntegrator<dim, 1, number> phi_basis;

      /// Face evaluator used to compute the gradient of the Lagrange interpolant of the primitive
      /// variables in the nodes.
      FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_primitive;

      /// Values in the face quadrature points.
      dealii::AlignedVector<VectorizedArrayType> basis_normal_derivative;
      dealii::AlignedVector<PrimitiveVariables>  primitive_m;
      dealii::AlignedVector<PrimitiveGradient>   gradient_m;
      dealii::AlignedVector<PrimitiveVariables>  normal_derivative_correction;
      dealii::AlignedVector<PrimitiveVariables>  boundary_correction;
      dealii::AlignedVector<ConservedVariables>  target;
    };

    /**
     * @brief Constructor.
     *
     * @param flow_scratch_data Reference to the flow scratch data object.
     */
    explicit NonReflectingBoundaryEvaluator(
      const OperationScratchData<dim, number> &flow_scratch_data)
      : flow_scratch_data(flow_scratch_data)
    {}

    /**
     * @brief Check whether any boundary of the domain is non-reflecting.
     */
    bool
    is_active() const
    {
      return not flow_scratch_data.boundary_conditions.get_non_reflecting_boundary_ids().empty();
    }

    /**
     * @brief Set up the internal data structures, i.e. the least-squares projection on the faces.
     *
     * Must be called after the matrix-free object has been set up. The boundary conditions may be
     * set later.
     */
    void
    reinit()
    {
      const auto &shape_data =
        flow_scratch_data.scratch_data.get_matrix_free()
          .get_shape_info(flow_scratch_data.dof_idx, flow_scratch_data.quad_idx)
          .data[0];

      fe_degree                = shape_data.fe_degree;
      nodal_at_cell_boundaries = shape_data.nodal_at_cell_boundaries;
      n_q_points_1d            = shape_data.n_q_points_1d;

      projection_1d.clear();
      if (fe_degree >= 1 and n_q_points_1d > fe_degree + 1)
        projection_1d = NonReflectingBoundary::least_squares_projection_matrix<number>(
          shape_data.quadrature.get_points(), fe_degree);

      is_initialized = true;
    }

    /**
     * @brief Throw an exception if the non-reflecting boundaries are used with unsupported settings.
     */
    void
    check_requirements() const
    {
      AssertThrow(is_initialized,
                  dealii::ExcMessage("The NonReflectingBoundaryEvaluator is not initialized."));
      AssertThrow(NonReflectingBoundary::is_supported_equation_of_state(flow_scratch_data.material),
                  dealii::ExcMessage("The non-reflecting boundary conditions are only implemented "
                                     "for ideal gases and single-species stiffened gases."));
      AssertThrow(fe_degree >= 1,
                  dealii::ExcMessage("The polynomial-correction non-reflecting boundary conditions "
                                     "require a polynomial degree of at least one."));
      AssertThrow(nodal_at_cell_boundaries,
                  dealii::ExcMessage("The polynomial-correction non-reflecting boundary conditions "
                                     "require a nodal basis with nodes on the element boundary."));
    }

    /**
     * @brief Compute the mean Mach number on each non-reflecting boundary.
     *
     * The mean Mach number enters the penalty terms of the incoming wave amplitudes, see Eqs. (24)
     * and (26) of the reference. Must be called before evaluate() whenever the solution changes.
     *
     * @param src The current solution.
     */
    void
    update_mean_mach_numbers(const VectorType &src) const
    {
      check_requirements();

      const std::vector<dealii::types::boundary_id> boundary_ids =
        flow_scratch_data.boundary_conditions.get_non_reflecting_boundary_ids();

      // integrals of the Mach number and of one for each boundary
      std::vector<number> integrals(2 * boundary_ids.size(), 0.);

      const auto &matrix_free = flow_scratch_data.scratch_data.get_matrix_free();
      FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi(
        matrix_free, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

      const unsigned int begin = matrix_free.n_inner_face_batches();
      const unsigned int end   = begin + matrix_free.n_boundary_face_batches();
      for (unsigned int face = begin; face < end; ++face)
        {
          const auto boundary =
            std::find(boundary_ids.begin(), boundary_ids.end(), matrix_free.get_boundary_id(face));
          if (boundary == boundary_ids.end())
            continue;
          const unsigned int index = std::distance(boundary_ids.begin(), boundary);

          phi.reinit(face);
          phi.gather_evaluate(src, dealii::EvaluationFlags::values);

          const unsigned int n_lanes = matrix_free.n_active_entries_per_face_batch(face);
          for (const unsigned int q : phi.quadrature_point_indices())
            {
              const ConservedVariables w = phi.get_value(q);
              const NSpeciesDofStateView<dim, n_species, number, const ConservedVariables> state(
                w, flow_scratch_data.material);
              const VectorizedArrayType mach = state.velocity().norm() / state.speed_of_sound();
              const VectorizedArrayType JxW  = phi.JxW(q);
              for (unsigned int v = 0; v < n_lanes; ++v)
                {
                  integrals[2 * index] += mach[v] * JxW[v];
                  integrals[2 * index + 1] += JxW[v];
                }
            }
        }

      std::vector<number> global_integrals(integrals.size());
      dealii::Utilities::MPI::sum(dealii::ArrayView<const number>(integrals),
                                  flow_scratch_data.scratch_data.get_mpi_comm(
                                    flow_scratch_data.dof_idx),
                                  dealii::ArrayView<number>(global_integrals));

      for (unsigned int i = 0; i < boundary_ids.size(); ++i)
        mean_mach_numbers[boundary_ids[i]] =
          global_integrals[2 * i + 1] > 0. ? global_integrals[2 * i] / global_integrals[2 * i + 1] :
                                             0.;
    }

    /**
     * @brief Get the mean Mach number on a non-reflecting boundary computed by the last call of
     * update_mean_mach_numbers().
     */
    number
    get_mean_mach_number(const dealii::types::boundary_id boundary_id) const
    {
      const auto it = mean_mach_numbers.find(boundary_id);
      return it == mean_mach_numbers.end() ? 0. : it->second;
    }

    /**
     * @brief Compute the boundary state and its gradient on a non-reflecting boundary face.
     *
     * As in Eqs. (27) and (A.13) of the reference, the interior trace and the derivatives of the
     * primitive variables are the ones of the Lagrange interpolant of the primitive variables in
     * the nodes. In contrast to evaluating the primitive variables of the interpolated conserved
     * variables, this gives a constant pressure at species interfaces in pressure equilibrium,
     * where the heat capacity ratio varies. Otherwise, the nonlinear equation of state would result
     * in a spurious pressure and pressure gradient, which the characteristic decomposition
     * partially turns into a correction of the boundary state.
     *
     * @param phi_m Face evaluator of the interior side, which is initialized to a boundary face of
     * a non-reflecting boundary and has evaluated the values and the gradients.
     * @param src The current solution, from which @p phi_m has been evaluated.
     * @param scratch Scratch data of the calling thread.
     * @param w_p On output, the boundary state in the face quadrature points.
     * @param grad_w_p On output, the gradient of the boundary state in the face quadrature points.
     */
    template <typename FaceEvaluator>
    void
    evaluate(const FaceEvaluator                               &phi_m,
             const VectorType                                  &src,
             FaceScratch                                       &scratch,
             dealii::AlignedVector<ConservedVariables>         &w_p,
             dealii::AlignedVector<ConservedVariablesGradient> &grad_w_p) const
    {
      using Idx            = NonReflectingBoundary::PrimitiveIndex<dim>;
      using StateView      = NSpeciesDofStateView<dim, n_species, number, const ConservedVariables>;
      const auto &material = flow_scratch_data.material;
      const auto &bcs      = flow_scratch_data.boundary_conditions;
      const auto &data     = flow_scratch_data.flow_data.non_reflecting_boundary;
      const auto  boundary_id = phi_m.boundary_id();
      const auto  n_q_points  = phi_m.n_q_points;

      NonReflectingBoundary::BoundaryParameters<dim, number, n_species> parameters;
      parameters.type = bcs.get_boundary_type(boundary_id);
      Assert(parameters.type == BoundaryConditionType::non_reflecting_inflow or
               parameters.type == BoundaryConditionType::non_reflecting_outflow,
             dealii::ExcMessage("The boundary is not a non-reflecting boundary."));
      const bool is_inflow = parameters.type == BoundaryConditionType::non_reflecting_inflow;
      parameters.sigma =
        is_inflow ? data.inflow_relaxation_coefficient : data.outflow_relaxation_coefficient;
      parameters.beta             = is_inflow ? 0. : data.outflow_transverse_relaxation_coefficient;
      parameters.reference_length = data.reference_length;
      parameters.mean_mach_number = get_mean_mach_number(boundary_id);

      compute_boundary_basis_normal_derivative(phi_m.get_current_cell_index(), scratch);
      evaluate_primitive_gradient(phi_m.get_current_cell_index(), src, scratch);

      auto &basis_normal_derivative      = scratch.basis_normal_derivative;
      auto &primitive_m                  = scratch.primitive_m;
      auto &gradient_m                   = scratch.gradient_m;
      auto &normal_derivative_correction = scratch.normal_derivative_correction;
      auto &boundary_correction          = scratch.boundary_correction;
      auto &target_m                     = scratch.target;

      primitive_m.resize_fast(n_q_points);
      gradient_m.resize_fast(n_q_points);
      normal_derivative_correction.resize_fast(n_q_points);
      boundary_correction.resize_fast(n_q_points);
      target_m.resize_fast(n_q_points);
      w_p.resize_fast(n_q_points);
      grad_w_p.resize_fast(n_q_points);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const ConservedVariables w_m = phi_m.get_value(q);
          const StateView          state(w_m, material);

          primitive_m[q] = scratch.phi_primitive.get_value(q);
          gradient_m[q]  = scratch.phi_primitive.get_gradient(q);

          // target state
          if (is_inflow)
            {
              ConservedVariables target;
              for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
                target[c] = bcs.get_boundary_value(boundary_id,
                                                   BoundaryConditionType::non_reflecting_inflow,
                                                   phi_m.quadrature_point(q),
                                                   c);
              target_m[q] = target;
              parameters.target =
                NonReflectingBoundary::primitive_variables<dim, n_species, number>(
                  StateView(target_m[q], material));
            }
          else
            parameters.target[Idx::pressure] =
              bcs.get_boundary_value(boundary_id,
                                     BoundaryConditionType::non_reflecting_outflow,
                                     phi_m.quadrature_point(q),
                                     NonReflectingOutflowValueInterpretation::pressure);

          normal_derivative_correction[q] =
            NonReflectingBoundary::normal_derivative_correction<dim, number, n_species>(
              primitive_m[q],
              gradient_m[q],
              phi_m.normal_vector(q),
              state.speed_of_sound(),
              basis_normal_derivative[q],
              parameters);

          // change of the face values which gives the boundary-conforming normal derivative while
          // the values in the interior of the element are fixed, Eq. (A.9)
          boundary_correction[q] = normal_derivative_correction[q] / basis_normal_derivative[q];
        }

      // least-squares fit of the face polynomial, Eqs. (A.11)-(A.16)
      if (not projection_1d.empty())
        NonReflectingBoundary::apply_face_projection<dim>(projection_1d,
                                                          n_q_points_1d,
                                                          boundary_correction);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const PrimitiveVariables primitive_p = primitive_m[q] + boundary_correction[q];
          w_p[q] = NonReflectingBoundary::conserved_variables<dim, n_species, number>(primitive_p,
                                                                                      material);

          // gradient of the boundary state with the boundary-conforming normal derivative
          const auto        normal     = phi_m.normal_vector(q);
          PrimitiveGradient gradient_p = gradient_m[q];
          for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
            gradient_p[c] += normal_derivative_correction[q][c] * normal;

          const ConservedVariables pressure_derivatives_p =
            NonReflectingBoundary::pressure_derivatives<dim, n_species, number>(
              StateView(w_p[q], material));
          grad_w_p[q] = NonReflectingBoundary::conserved_gradient<dim, n_species, number>(
            w_p[q], gradient_p, pressure_derivatives_p);

          if (is_inflow)
            {
              // At supersonic inflow, all waves enter the domain and the prescribed state is
              // imposed.
              VectorizedArrayType normal_velocity = 0.;
              for (unsigned int d = 0; d < dim; ++d)
                normal_velocity += primitive_m[q][Idx::velocity + d] * normal[d];
              const VectorizedArrayType speed_of_sound =
                StateView(phi_m.get_value(q), material).speed_of_sound();
              const ConservedVariablesGradient grad_w_m = phi_m.get_gradient(q);
              for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
                {
                  w_p[q][c] =
                    dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
                      normal_velocity + speed_of_sound,
                      VectorizedArrayType(0.),
                      target_m[q][c],
                      w_p[q][c]);
                  for (unsigned int d = 0; d < dim; ++d)
                    grad_w_p[q][c][d] =
                      dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
                        normal_velocity + speed_of_sound,
                        VectorizedArrayType(0.),
                        grad_w_m[c][d],
                        grad_w_p[q][c][d]);
                }
            }
        }
    }

  private:
    /// Scratch data of the compressible flow solver.
    const OperationScratchData<dim, number> &flow_scratch_data;

    /// Whether reinit() has been called.
    bool is_initialized = false;

    /// Polynomial degree of the finite element.
    unsigned int fe_degree = 0;

    /// Whether the basis is nodal on the element boundary.
    bool nodal_at_cell_boundaries = false;

    /// Number of quadrature points in one dimension.
    unsigned int n_q_points_1d = 0;

    /// One-dimensional least-squares projection on the face quadrature points. Empty if the number
    /// of quadrature points equals the number of nodes, such that the projection is the identity.
    std::vector<number> projection_1d;

    /// Mean Mach number of each non-reflecting boundary.
    mutable std::map<dealii::types::boundary_id, number> mean_mach_numbers;

    /**
     * @brief Evaluate the gradient of the Lagrange interpolant of the primitive variables in the
     * nodes at the face quadrature points.
     *
     * @param face Index of the face batch.
     * @param src The current solution.
     * @param scratch Scratch data, whose evaluator phi_primitive holds the gradients on output.
     */
    void
    evaluate_primitive_gradient(const unsigned int face,
                                const VectorType  &src,
                                FaceScratch       &scratch) const
    {
      constexpr unsigned int n_components = n_conserved_variables<dim, n_species>;

      auto &phi = scratch.phi_primitive;
      phi.reinit(face);
      phi.read_dof_values(src);

      // The basis is nodal, so the DoF values are the values in the nodes.
      VectorizedArrayType *dof_values         = phi.begin_dof_values();
      const unsigned int   dofs_per_component = phi.dofs_per_component;
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        {
          ConservedVariables w;
          for (unsigned int c = 0; c < n_components; ++c)
            w[c] = dof_values[c * dofs_per_component + i];

          const PrimitiveVariables primitive =
            NonReflectingBoundary::primitive_variables<dim, n_species, number>(
              NSpeciesDofStateView<dim, n_species, number, const ConservedVariables>(
                w, flow_scratch_data.material));

          for (unsigned int c = 0; c < n_components; ++c)
            dof_values[c * dofs_per_component + i] = primitive[c];
        }

      phi.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
    }

    /**
     * @brief Compute the normal derivative of the Lagrange polynomial of the face nodes in the face
     * quadrature points.
     *
     * The function which is one in all face nodes and zero in all other nodes is the
     * one-dimensional Lagrange polynomial of the face nodes in the direction normal to the face,
     * since the Lagrange polynomials in the tangential directions sum up to one. Its normal
     * derivative is the factor \f$c\f$ relating the face values to their contribution to the normal
     * derivative, which includes the effect of the mapping.
     *
     * @param face Index of the face batch.
     * @param scratch Scratch data, whose field basis_normal_derivative is filled.
     */
    void
    compute_boundary_basis_normal_derivative(const unsigned int face, FaceScratch &scratch) const
    {
      auto &phi_basis = scratch.phi_basis;
      phi_basis.reinit(face);

      const unsigned int   face_no    = phi_basis.get_face_no();
      const unsigned int   direction  = face_no / 2;
      const unsigned int   n_dofs_1d  = phi_basis.get_shape_info().data[0].fe_degree + 1;
      const unsigned int   face_index = (face_no % 2 == 0) ? 0 : n_dofs_1d - 1;
      const unsigned int   stride     = dealii::Utilities::pow(n_dofs_1d, direction);
      VectorizedArrayType *dof_values = phi_basis.begin_dof_values();
      for (unsigned int i = 0; i < phi_basis.dofs_per_cell; ++i)
        dof_values[i] = ((i / stride) % n_dofs_1d == face_index) ? 1. : 0.;

      phi_basis.evaluate(dealii::EvaluationFlags::gradients);

      scratch.basis_normal_derivative.resize_fast(phi_basis.n_q_points);
      for (const unsigned int q : phi_basis.quadrature_point_indices())
        scratch.basis_normal_derivative[q] = phi_basis.get_normal_derivative(q);
    }
  };
} // namespace MeltPoolDG::CompressibleFlow
