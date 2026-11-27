#pragma once

#include <deal.II/base/exceptions.h>

#include <meltpooldg/compressible_flow/boundary_state_evaluation.hpp>
#include <meltpooldg/species_transport/boundary_conditions.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <limits>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number, int n_species>
  BoundaryStateEvaluation<dim, number, n_species>::BoundaryStateEvaluation(
    const BoundaryConditions<dim, number> &boundary_conditions,
    const unsigned int                     n_points)
    : boundary_conditions(boundary_conditions)
    , exterior_values(n_points)
    , exterior_gradients(n_points)
    , n_evaluated_points(0)
    , evaluated_flags(dealii::EvaluationFlags::nothing)
  {}

  template <int dim, typename number, int n_species>
  template <IsBoundaryStateView<dim, number, n_species> ReadView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::evaluate(
    const FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> &phi_m,
    const MaterialPhaseData<number>                                            &material,
    const dealii::EvaluationFlags::EvaluationFlags                              evaluation_flags,
    const bool use_mirror_principle)
  {
    Assert(evaluation_flags == dealii::EvaluationFlags::values or
             evaluation_flags ==
               (dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients),
           dealii::ExcMessage("BoundaryStateEvaluation::evaluate() only accepts "
                              "EvaluationFlags::values or "
                              "EvaluationFlags::values | EvaluationFlags::gradients."));
    Assert(phi_m.n_q_points <= exterior_values.size(),
           dealii::ExcMessage("The number of quadrature points of the face batch exceeds the "
                              "number of points for which the buffers were allocated."));

    n_evaluated_points = phi_m.n_q_points;
    evaluated_flags    = evaluation_flags;

    // The non-reflecting outflow needs the data of the whole face, so it is not evaluated point by
    // point.
    if (boundary_conditions.get_boundary_type(phi_m.boundary_id()) ==
        BoundaryConditionType::subsonic_non_reflecting_pressure_outflow)
      {
        evaluate_non_reflecting_outflow<ReadView>(phi_m, material, evaluation_flags);
        return;
      }

    // All lanes of a boundary face batch lie on the same boundary. This is guaranteed by
    // dealii::MatrixFree, which only puts boundary faces with the same boundary id into one batch.
    std::bitset<VectorizedArrayType::size()> all_lanes;
    all_lanes.set();

    for (const unsigned int q : phi_m.quadrature_point_indices())
      {
        evaluate_point<ReadView>(q,
                                 phi_m.boundary_id(),
                                 all_lanes,
                                 phi_m.quadrature_point(q),
                                 phi_m.normal_vector(q),
                                 phi_m.get_value(q),
                                 phi_m.get_gradient(q),
                                 material,
                                 evaluation_flags,
                                 use_mirror_principle);
      }
  }

  template <int dim, typename number, int n_species>
  template <IsBoundaryStateView<dim, number, n_species> ReadView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::evaluate(
    const std::array<dealii::types::boundary_id, VectorizedArrayType::size()>     &boundary_ids,
    const dealii::ArrayView<const dealii::Point<dim, VectorizedArrayType>>        &locations,
    const dealii::ArrayView<const dealii::Tensor<1, dim, VectorizedArrayType>>    &normals,
    const dealii::ArrayView<const ConservedVariablesType<dim, number, n_species>> &interior_values,
    const dealii::ArrayView<const ConservedVariablesGradientType<dim, number, n_species>>
                                                  &interior_gradients,
    const MaterialPhaseData<number>               &material,
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
    const bool                                     use_mirror_principle)
  {
    constexpr unsigned int n_lanes  = VectorizedArrayType::size();
    const unsigned int     n_points = interior_values.size();

    Assert(evaluation_flags == dealii::EvaluationFlags::values or
             evaluation_flags ==
               (dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients),
           dealii::ExcMessage("BoundaryStateEvaluation::evaluate() only accepts "
                              "EvaluationFlags::values or "
                              "EvaluationFlags::values | EvaluationFlags::gradients."));
    Assert(n_points <= exterior_values.size(),
           dealii::ExcMessage("The number of points of the batch exceeds the number of points for "
                              "which the buffers were allocated."));
    AssertDimension(locations.size(), n_points);
    AssertDimension(normals.size(), n_points);
    Assert(interior_gradients.empty() or interior_gradients.size() == n_points,
           dealii::ExcMessage("The interior gradients must be empty or given at all points."));

    n_evaluated_points = n_points;
    evaluated_flags    = evaluation_flags;

    // The existing boundary types do not read the interior gradient for the exterior values.
    const ConservedVariablesGradientType<dim, number, n_species> zero_gradient{};

    // Keep track of which lanes have already been handled, as the boundary ids may be different for
    // each lane of the batch.
    std::bitset<n_lanes> handled_lanes;
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        if (handled_lanes[lane] or boundary_ids[lane] == dealii::numbers::invalid_boundary_id)
          continue;

        const dealii::types::boundary_id boundary_id = boundary_ids[lane];

        std::bitset<n_lanes> lanes_of_boundary;
        for (unsigned int other_lane = lane; other_lane < n_lanes; ++other_lane)
          if (boundary_ids[other_lane] == boundary_id)
            lanes_of_boundary.set(other_lane);
        handled_lanes |= lanes_of_boundary;

        for (unsigned int q = 0; q < n_points; ++q)
          evaluate_point<ReadView>(q,
                                   boundary_id,
                                   lanes_of_boundary,
                                   locations[q],
                                   normals[q],
                                   interior_values[q],
                                   interior_gradients.empty() ? zero_gradient :
                                                                interior_gradients[q],
                                   material,
                                   evaluation_flags,
                                   use_mirror_principle);
      }
  }

  template <int dim, typename number, int n_species>
  const ConservedVariablesType<dim, number, n_species> &
  BoundaryStateEvaluation<dim, number, n_species>::get_value(const unsigned int q) const
  {
    AssertIndexRange(q, n_evaluated_points);
    return exterior_values[q];
  }

  template <int dim, typename number, int n_species>
  const ConservedVariablesGradientType<dim, number, n_species> &
  BoundaryStateEvaluation<dim, number, n_species>::get_gradient(const unsigned int q) const
  {
    Assert(evaluated_flags & dealii::EvaluationFlags::gradients,
           dealii::ExcMessage("The exterior gradients were not requested in the last call of "
                              "BoundaryStateEvaluation::evaluate()."));
    AssertIndexRange(q, n_evaluated_points);
    return exterior_gradients[q];
  }

  template <int dim, typename number, int n_species>
  template <typename ReadView, typename WriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::compute_exterior_state(
    const BoundaryConditions<dim, number>             &boundary_conditions,
    const dealii::Point<dim, VectorizedArrayType>     &location,
    const dealii::Tensor<1, dim, VectorizedArrayType> &normal,
    const dealii::types::boundary_id                   boundary_id,
    const ReadView                                    &w_m,
    const WriteView                                   &w_p,
    const bool                                         use_mirror_principle)
  {
    if (const BoundaryConditionType boundary_type =
          boundary_conditions.get_boundary_type(boundary_id);
        boundary_type == BoundaryConditionType::slip_wall)
      {
        apply_slip_wall_boundary(w_m, w_p, normal, use_mirror_principle);
      }
    else if (boundary_type == BoundaryConditionType::no_slip_wall)
      {
        apply_no_slip_wall_boundary(w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::inflow)
      {
        apply_inflow_boundary(boundary_conditions, boundary_id, location, w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::subsonic_outflow_fixed_pressure)
      {
        apply_subsonic_outflow_with_fixed_pressure_boundary(
          boundary_conditions, boundary_id, location, w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::subsonic_outflow_fixed_energy)
      {
        apply_subsonic_outflow_with_fixed_energy_boundary(
          boundary_conditions, boundary_id, location, w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::combined_inflow_no_slip_wall)
      {
        apply_combined_inflow_no_slip_wall_boundary(
          boundary_conditions, boundary_id, location, w_m, w_p);
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage(
                      std::format("Unknown boundary condition {} at the boundary with boundary id "
                                  "{}. Did you set a boundary condition for this part of the "
                                  "domain boundary?",
                                  boundary_type._to_string(),
                                  boundary_id)));
      }
  }

  template <int dim, typename number, int n_species>
  template <typename ReadView, typename WriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::compute_exterior_partial_densities(
    const BoundaryConditions<dim, number>         &boundary_conditions,
    const dealii::Point<dim, VectorizedArrayType> &location,
    const dealii::types::boundary_id               boundary_id,
    const ReadView                                &w_m,
    const WriteView                               &w_p)
  {
    if (const BoundaryConditionType boundary_type =
          boundary_conditions.get_boundary_type(boundary_id);
        boundary_type == BoundaryConditionType::inflow)
      {
        // For inflow boundary conditions we assume that the partial densities are given and we set
        // them as Dirichlet values.

        // Remember for compressible flow we only need to prescribe n_species - 1 partial densities.
        std::array<VectorizedArrayType, n_species - 1> dirichlet_boundary_values;

        for (unsigned int species = 0; species < n_species - 1; ++species)
          {
            dirichlet_boundary_values[species] = boundary_conditions.get_boundary_value(
              boundary_id,
              BoundaryConditionType::inflow,
              location,
              CompressibleFlow::n_conserved_variables<dim, 1> + species);
          }

        SpeciesTransport::set_boundary_value_and_gradient(
          w_p, w_m, SpeciesTransport::BoundaryConditionType::dirichlet, dirichlet_boundary_values);
      }
    else if (boundary_type == BoundaryConditionType::combined_inflow_no_slip_wall)
      {
        VectorizedArrayType mask = boundary_conditions.get_boundary_value(
          boundary_id,
          BoundaryConditionType::combined_inflow_no_slip_wall,
          location,
          CombinedInflowNoSlipWallValueInterpretation<dim>::inside_flow);

        std::array<VectorizedArrayType, n_species - 1> dirichlet_boundary_values;
        for (unsigned int species = 0; species < n_species - 1; ++species)
          {
            dirichlet_boundary_values[species] = boundary_conditions.get_boundary_value(
              boundary_id,
              BoundaryConditionType::combined_inflow_no_slip_wall,
              location,
              CombinedInflowNoSlipWallValueInterpretation<dim>::mass_fractions + species);
          }

        SpeciesTransport::set_boundary_value_and_gradient(
          w_p, w_m, SpeciesTransport::BoundaryConditionType::dirichlet, dirichlet_boundary_values);

        std::vector<VectorizedArrayType> w_p_partial_densities(n_species - 1);
        for (unsigned int species = 0; species < n_species - 1; ++species)
          w_p_partial_densities[species] = w_p.partial_density(species);

        std::vector<dealii::Tensor<1, dim, VectorizedArrayType>> w_p_partial_density_gradients(
          n_species - 1);
        for (unsigned int species = 0; species < n_species - 1; ++species)
          w_p_partial_density_gradients[species] = w_p.grad_partial_density(species);

        SpeciesTransport::
          set_boundary_value_and_gradient<n_species - 1, VectorizedArrayType, ReadView, WriteView>(
            w_p, w_m, SpeciesTransport::BoundaryConditionType::neumann);

        for (unsigned int species = 0; species < n_species - 1; ++species)
          {
            w_p.partial_density(species) =
              dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                mask,
                VectorizedArrayType(1.0),
                w_p_partial_densities[species],
                w_p.partial_density(species));

            for (unsigned int d = 0; d < dim; ++d)
              w_p.grad_partial_density(species)[d] =
                dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                  mask,
                  VectorizedArrayType(1.0),
                  w_p_partial_density_gradients[species][d],
                  w_p.grad_partial_density(species)[d]);
          }
      }
    else
      {
        // For all other boundary conditions we assume homogeneous Neumann for the partial
        // densities.
        SpeciesTransport::
          set_boundary_value_and_gradient<n_species - 1, VectorizedArrayType, ReadView, WriteView>(
            w_p, w_m, SpeciesTransport::BoundaryConditionType::neumann);
      }
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::apply_slip_wall_boundary(
    DofReadView                                        w_m,
    DofWriteView                                       w_p,
    const dealii::Tensor<1, dim, VectorizedArrayType> &normal,
    const bool                                         use_mirror_principle)
  {
    // homogeneous Neumann
    w_p.density()      = w_m.density();
    w_p.grad_density() = -(w_m.grad_density());

    // homogeneous Neumann
    VectorizedArrayType rho_u_dot_n = 0;
    for (unsigned int d = 0; d < dim; ++d)
      rho_u_dot_n += w_m.momentum(d) * normal[d];
    // symmetry: the mirror principle reverses the normal momentum, while if no mirror value is used
    // the value on the boundary has no normal momentum
    const number normal_momentum_factor = use_mirror_principle ? 2. : 1.;
    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d) = w_m.momentum(d) - normal_momentum_factor * rho_u_dot_n * normal[d];
        w_p.grad_momentum(d) =
          w_m.grad_momentum(d) - 2. * scalar_product(w_m.grad_momentum(d), normal) * normal;
      }
    // homogeneous Neumann
    w_p.grad_total_energy() = -(w_m.grad_total_energy());
    w_p.total_energy()      = w_m.total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::apply_no_slip_wall_boundary(DofReadView  w_m,
                                                                               DofWriteView w_p)
  {
    // homogeneous Neumann
    w_p.density()      = w_m.density();
    w_p.grad_density() = -(w_m.grad_density());

    // Dirichlet
    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d)      = 0.;
        w_p.grad_momentum(d) = w_m.grad_momentum(d);
      }
    // homogeneous Neumann
    w_p.grad_total_energy() = -(w_m.grad_total_energy());
    w_p.total_energy()      = w_m.total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView, typename DirichletValueInterpretation>
  void
  BoundaryStateEvaluation<dim, number, n_species>::apply_inflow_boundary(
    const BoundaryConditions<dim, number>         &boundary_conditions,
    const dealii::types::boundary_id               boundary_id,
    const dealii::Point<dim, VectorizedArrayType> &q_point,
    DofReadView                                    w_m,
    DofWriteView                                   w_p,
    const BoundaryConditionType                    bc_value_type)
  {
    w_p.density()      = boundary_conditions.get_boundary_value(boundary_id,
                                                           bc_value_type,
                                                           q_point,
                                                           DirichletValueInterpretation::density);
    w_p.grad_density() = w_m.grad_density();

    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d) = boundary_conditions.get_boundary_value(
          boundary_id, bc_value_type, q_point, DirichletValueInterpretation::velocity + d);
        w_p.grad_momentum(d) = w_m.grad_momentum(d);
      }

    w_p.total_energy() = boundary_conditions.get_boundary_value(
      boundary_id, bc_value_type, q_point, DirichletValueInterpretation::energy);
    w_p.grad_total_energy() = w_m.grad_total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::
    apply_subsonic_outflow_with_fixed_pressure_boundary(
      const BoundaryConditions<dim, number>         &boundary_conditions,
      const dealii::types::boundary_id               boundary_id,
      const dealii::Point<dim, VectorizedArrayType> &q_point,
      DofReadView                                    w_m,
      DofWriteView                                   w_p)
  {
    // homogeneous Neumann
    w_p.density()      = w_m.density();
    w_p.grad_density() = -(w_m.grad_density());

    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d)      = w_m.momentum(d);
        w_p.grad_momentum(d) = -(w_m.grad_momentum(d));
      }

    // Dirichlet
    auto p_dyn = VectorizedArrayType(0.);
    for (unsigned int i = 0; i < dim; ++i)
      p_dyn += w_m.momentum(i) * w_m.momentum(i);

    p_dyn /= (w_m.density() * 2.);
    const VectorizedArrayType pressure = boundary_conditions.get_boundary_value(
      boundary_id, BoundaryConditionType::subsonic_outflow_fixed_pressure, q_point, 0);

    // consider equation of state for computation of inner energy from given pressure
    const VectorizedArrayType inner_energy = w_m.inner_energy_from_pressure(pressure);

    w_p.total_energy()      = inner_energy + p_dyn;
    w_p.grad_total_energy() = w_m.grad_total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::
    apply_subsonic_outflow_with_fixed_energy_boundary(
      const BoundaryConditions<dim, number>         &boundary_conditions,
      const dealii::types::boundary_id               boundary_id,
      const dealii::Point<dim, VectorizedArrayType> &q_point,
      DofReadView                                    w_m,
      DofWriteView                                   w_p)
  {
    // homogeneous Neumann
    w_p.density()      = w_m.density();
    w_p.grad_density() = -(w_m.grad_density());
    for (unsigned int i = 0; i < dim; ++i)
      {
        w_p.momentum(i)      = w_m.momentum(i);
        w_p.grad_momentum(i) = -(w_m.grad_momentum(i));
      }

    // Dirichlet
    w_p.total_energy() = boundary_conditions.get_boundary_value(
      boundary_id, BoundaryConditionType::subsonic_outflow_fixed_energy, q_point, 0);
    w_p.grad_total_energy() = w_m.grad_total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::apply_combined_inflow_no_slip_wall_boundary(
    const BoundaryConditions<dim, number>         &boundary_conditions,
    const dealii::types::boundary_id               boundary_id,
    const dealii::Point<dim, VectorizedArrayType> &q_point,
    DofReadView                                    w_m,
    DofWriteView                                   w_p)
  {
    VectorizedArrayType is_inflow = boundary_conditions.get_boundary_value(
      boundary_id,
      BoundaryConditionType::combined_inflow_no_slip_wall,
      q_point,
      CombinedInflowNoSlipWallValueInterpretation<dim>::inside_flow);

    apply_inflow_boundary<DofReadView,
                          DofWriteView,
                          CombinedInflowNoSlipWallValueInterpretation<dim>>(
      boundary_conditions,
      boundary_id,
      q_point,
      w_m,
      w_p,
      BoundaryConditionType::combined_inflow_no_slip_wall);

    auto inflow_density  = w_p.density();
    auto inflow_momentum = w_p.momentum();
    auto inflow_energy   = w_p.total_energy();

    auto inflow_grad_density  = w_p.grad_density();
    auto inflow_grad_momentum = w_p.grad_momentum();
    auto inflow_grad_energy   = w_p.grad_total_energy();

    apply_no_slip_wall_boundary(w_m, w_p);

    auto apply_mask = [](auto mask, auto true_value, auto false_value) -> auto {
      return dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(mask,
                                                                           VectorizedArrayType(1.0),
                                                                           true_value,
                                                                           false_value);
    };

    w_p.density() = apply_mask(is_inflow, inflow_density, w_p.density());
    for (unsigned int d = 0; d < dim; ++d)
      w_p.grad_density()[d] = apply_mask(is_inflow, inflow_grad_density[d], w_p.grad_density()[d]);

    w_p.total_energy() = apply_mask(is_inflow, inflow_energy, w_p.total_energy());
    for (unsigned int d = 0; d < dim; ++d)
      w_p.grad_total_energy()[d] =
        apply_mask(is_inflow, inflow_grad_energy[d], w_p.grad_total_energy()[d]);

    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d) = apply_mask(is_inflow, inflow_momentum[d], w_p.momentum(d));
        for (unsigned int d2 = 0; d2 < dim; ++d2)
          w_p.grad_momentum(d)[d2] =
            apply_mask(is_inflow, inflow_grad_momentum[d][d2], w_p.grad_momentum(d)[d2]);
      }
  }

  template <int dim, typename number, int n_species>
  template <typename ReadView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::evaluate_point(
    const unsigned int                                            q,
    const dealii::types::boundary_id                              boundary_id,
    const std::bitset<VectorizedArrayType::size()>               &lanes,
    const dealii::Point<dim, VectorizedArrayType>                &location,
    const dealii::Tensor<1, dim, VectorizedArrayType>            &normal,
    const ConservedVariablesType<dim, number, n_species>         &interior_value,
    const ConservedVariablesGradientType<dim, number, n_species> &interior_gradient,
    const MaterialPhaseData<number>                              &material,
    const dealii::EvaluationFlags::EvaluationFlags                evaluation_flags,
    const bool                                                    use_mirror_principle)
  {
    ConservedVariablesType<dim, number, n_species>         exterior_value;
    ConservedVariablesGradientType<dim, number, n_species> exterior_gradient;

    const ReadView interior(interior_value, interior_gradient, material);
    const typename ReadView::template rebind<typename ReadView::state_type,
                                             typename ReadView::gradient_type>
      exterior(exterior_value, exterior_gradient, material);

    compute_exterior_state(
      boundary_conditions, location, normal, boundary_id, interior, exterior, use_mirror_principle);

    if constexpr (n_species > 1)
      compute_exterior_partial_densities(
        boundary_conditions, location, boundary_id, interior, exterior);

    // Copy only the lanes on this boundary.
    if (lanes.all())
      {
        exterior_values[q] = exterior_value;
        if (evaluation_flags & dealii::EvaluationFlags::gradients)
          exterior_gradients[q] = exterior_gradient;
      }
    else
      {
        const VectorizedArrayType lane_mask = bitset_to_vectorized_mask<number>(lanes);
        const VectorizedArrayType one       = 1.;

        for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
          {
            exterior_values[q][c] = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              lane_mask, one, exterior_value[c], exterior_values[q][c]);
            if (evaluation_flags & dealii::EvaluationFlags::gradients)
              for (unsigned int d = 0; d < dim; ++d)
                exterior_gradients[q][c][d] =
                  dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                    lane_mask, one, exterior_gradient[c][d], exterior_gradients[q][c][d]);
          }
      }
  }

  template <int dim, typename number, int n_species>
  template <typename ReadView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::evaluate_non_reflecting_outflow(
    const FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> &phi_m,
    const MaterialPhaseData<number>                                            &material,
    const dealii::EvaluationFlags::EvaluationFlags                              evaluation_flags)
  {
    const unsigned int degree = phi_m.get_shape_info().data[0].fe_degree;
    Assert(phi_m.get_shape_info().data[0].nodal_at_cell_boundaries,
           dealii::ExcMessage("The non-reflecting pressure outflow requires a basis that is nodal "
                              "at the cell boundaries, e.g. that of FE_DGQ."));
    Assert(degree >= 1,
           dealii::ExcMessage(
             "The non-reflecting pressure outflow requires a polynomial degree of at least 1."));
    Assert(phi_m.get_shape_info().data[0].n_q_points_1d == degree + 1,
           dealii::ExcMessage(
             std::format("The non-reflecting pressure outflow requires N + 1 = {} quadrature "
                         "points per direction, but the face quadrature has {}.",
                         degree + 1,
                         phi_m.get_shape_info().data[0].n_q_points_1d)));

    // Pass 1 computes the face-averaged normal Mach number of each lane. The Mach number only needs
    // the interior values, so the views get a zero gradient.
    const ConservedVariablesGradientType<dim, number, n_species> zero_gradient{};
    VectorizedArrayType                                          mach_number_integral = 0.;
    VectorizedArrayType                                          face_area            = 0.;
    for (const unsigned int q : phi_m.quadrature_point_indices())
      {
        const ConservedVariablesType<dim, number, n_species> interior_value = phi_m.get_value(q);
        const ReadView interior(interior_value, zero_gradient, material);

        mach_number_integral +=
          phi_m.JxW(q) * (interior.velocity() * phi_m.normal_vector(q)) / interior.speed_of_sound();
        face_area += phi_m.JxW(q);
      }
    const VectorizedArrayType mean_mach_number = mach_number_integral / face_area;

    // Precompute some values
    const dealii::Tensor<1, dim, VectorizedArrayType> reference_normal_derivative =
      phi_m.normal_vector(0) * phi_m.inverse_jacobian(0);
    const VectorizedArrayType delta =
      2. / (static_cast<number>(degree * (degree + 1)) * reference_normal_derivative.norm());


    const typename BoundaryConditionData<number>::NonReflectingOutflow &coefficients =
      boundary_conditions.get_boundary_condition_data().non_reflecting_outflow;
    const VectorizedArrayType relaxation_factor = coefficients.relaxation_coefficient *
                                                  (1. - mean_mach_number * mean_mach_number) /
                                                  (2. * coefficients.reference_length);

    // Pass 2 computes the exterior value at every quadrature point and, if requested, the exterior
    // gradient with the corrected normal derivative.
    for (const unsigned int q : phi_m.quadrature_point_indices())
      {
        // Read interior values and gradients at the quadrature point.
        const ConservedVariablesType<dim, number, n_species> interior_value = phi_m.get_value(q);
        const ConservedVariablesGradientType<dim, number, n_species> interior_gradient =
          phi_m.get_gradient(q);
        const ReadView interior(interior_value, interior_gradient, material);

        const dealii::Tensor<1, dim, VectorizedArrayType> normal   = phi_m.normal_vector(q);
        const VectorizedArrayType                         density  = interior.density();
        const dealii::Tensor<1, dim, VectorizedArrayType> velocity = interior.velocity();
        const VectorizedArrayType                         pressure = interior.pressure();
        const VectorizedArrayType speed_of_sound                   = interior.speed_of_sound();
        const dealii::Tensor<1, dim, dealii::Tensor<1, dim, VectorizedArrayType>>
          velocity_gradient = interior.grad_velocity();
        const dealii::Tensor<1, dim, VectorizedArrayType> pressure_gradient =
          interior.grad_pressure();

        // Compute normal and tangential components of the velocity and its derivatives.
        const VectorizedArrayType                         normal_velocity = velocity * normal;
        const dealii::Tensor<1, dim, VectorizedArrayType> tangential_velocity =
          velocity - normal_velocity * normal;

        dealii::Tensor<1, dim, VectorizedArrayType> normal_derivative_of_velocity;
        dealii::Tensor<1, dim, VectorizedArrayType> tangential_derivative_of_velocity;
        VectorizedArrayType                         velocity_divergence = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          {
            normal_derivative_of_velocity[d]     = velocity_gradient[d] * normal;
            tangential_derivative_of_velocity[d] = velocity_gradient[d] * tangential_velocity;
            velocity_divergence += velocity_gradient[d][d];
          }

        // Compute the interior amplitude and the transverse term
        const VectorizedArrayType inverse_impedance = 1. / (2. * density * speed_of_sound);

        const VectorizedArrayType normal_derivative_of_normal_velocity =
          normal * normal_derivative_of_velocity;
        const VectorizedArrayType interior_amplitude =
          (normal_velocity - speed_of_sound) * (inverse_impedance * (pressure_gradient * normal) -
                                                0.5 * normal_derivative_of_normal_velocity);

        // Compute the prescribed exterior amplitude
        const VectorizedArrayType transverse_term =
          inverse_impedance * (tangential_velocity * pressure_gradient) -
          0.5 * (normal * tangential_derivative_of_velocity) +
          0.5 * speed_of_sound * (velocity_divergence - normal_derivative_of_normal_velocity);

        const VectorizedArrayType target_pressure = boundary_conditions.get_boundary_value(
          phi_m.boundary_id(),
          BoundaryConditionType::subsonic_non_reflecting_pressure_outflow,
          phi_m.quadrature_point(q),
          0);
        const VectorizedArrayType exterior_amplitude =
          relaxation_factor * (pressure - target_pressure) / density -
          coefficients.transverse_coefficient * transverse_term;

        // The factor alpha by which the corrected normal derivative differs from the interior one
        // along the eigenvector r_1 of the incoming acoustic wave, and the jump Q^+ - Q^- =
        // delta alpha r_1.
        const VectorizedArrayType normal_derivative_correction =
          (exterior_amplitude - interior_amplitude) / (normal_velocity - speed_of_sound);
        const VectorizedArrayType jump_amplitude = delta * normal_derivative_correction;

        // Compute the exterior primitive variables Q^+
        const VectorizedArrayType exterior_density =
          density + jump_amplitude * density / speed_of_sound;
        const dealii::Tensor<1, dim, VectorizedArrayType> exterior_velocity =
          velocity - jump_amplitude * normal;
        const VectorizedArrayType exterior_pressure =
          pressure + jump_amplitude * density * speed_of_sound;

        // Convert the exterior primitive variables to conserved variables
        ConservedVariablesType<dim, number, n_species>         exterior_value;
        ConservedVariablesGradientType<dim, number, n_species> exterior_gradient;
        const typename ReadView::template rebind<typename ReadView::state_type,
                                                 typename ReadView::gradient_type>
          exterior(exterior_value, exterior_gradient, material);

        exterior.density() = exterior_density;
        for (unsigned int d = 0; d < dim; ++d)
          exterior.momentum(d) = exterior_density * exterior_velocity[d];
        if constexpr (n_species > 1)
          for (unsigned int species = 0; species < n_species - 1; ++species)
            exterior.partial_density(species) = exterior_density * interior.mass_fraction(species);
        exterior.total_energy() = exterior.inner_energy_from_pressure(exterior_pressure) +
                                  0.5 * exterior_density * (exterior_velocity * exterior_velocity);

        exterior_values[q] = exterior_value;

        // To compute the exterior gradient, we start from the interior gradient and correct it such
        // that we get the desired exterior gradient. This contains of two parts: (1) the correction
        // of the normal derivative such that the exterior amplitude is achieved, and (2) the
        // tangential gradient which should equal the tangential gradient of U^+.

        // Step (1): Apply the correction of the normal derivative to the interior gradient. The
        // normal derivative
        if (evaluation_flags & dealii::EvaluationFlags::gradients)
          {
            exterior_gradient = interior_gradient;

            const VectorizedArrayType scaled_correction =
              normal_derivative_correction * density / speed_of_sound;
            const VectorizedArrayType total_enthalpy =
              (interior.total_energy() + pressure) / density;

            exterior.grad_density() += scaled_correction * normal;
            for (unsigned int d = 0; d < dim; ++d)
              exterior.grad_momentum(d) +=
                scaled_correction * (velocity[d] - speed_of_sound * normal[d]) * normal;
            exterior.grad_total_energy() +=
              scaled_correction * (total_enthalpy - speed_of_sound * normal_velocity) * normal;
            if constexpr (n_species > 1)
              for (unsigned int species = 0; species < n_species - 1; ++species)
                exterior.grad_partial_density(species) +=
                  scaled_correction * interior.mass_fraction(species) * normal;

            exterior_gradients[q] = exterior_gradient;
          }
      }

    if (not(evaluation_flags & dealii::EvaluationFlags::gradients))
      return;

    // Pass 3
    const dealii::internal::MatrixFreeFunctions::UnivariateShapeData<number> &shape_data =
      phi_m.get_shape_info().data[0];
    const unsigned int n_q_points_1d = shape_data.n_q_points_1d;

    const std::function<unsigned int(const unsigned int)> face_dir_to_space_dir =
      [&phi_m](const unsigned int face_dir) -> unsigned int {
      const unsigned int normal_axis = phi_m.get_face_no() / 2;
      return (normal_axis + 1 + face_dir) % dim;
    };

    for (const unsigned int q : phi_m.quadrature_point_indices())
      {
        for (unsigned int face_direction = 0; face_direction < dim - 1; ++face_direction)
          {
            const unsigned int stride = dealii::Utilities::pow(n_q_points_1d, face_direction);
            const unsigned int position_of_q = (q / stride) % n_q_points_1d;
            const unsigned int first_point   = q - position_of_q * stride;

            ConservedVariablesType<dim, number, n_species> derivative_of_jump;
            for (unsigned int i = 0; i < n_q_points_1d; ++i)
              {
                const unsigned int point_on_line = first_point + i * stride;
                derivative_of_jump +=
                  shape_data.shape_gradients_collocation[i * n_q_points_1d + position_of_q] *
                  (exterior_values[point_on_line] - phi_m.get_value(point_on_line));
              }

            const unsigned int face_dir = face_dir_to_space_dir(face_direction);
            for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
              exterior_gradients[q][c][face_dir] +=
                phi_m.inverse_jacobian(0)[face_dir][face_direction] * derivative_of_jump[c];
          }
      }
  }
} // namespace MeltPoolDG::CompressibleFlow
