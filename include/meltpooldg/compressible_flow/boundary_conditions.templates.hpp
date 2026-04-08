#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/species_transport/boundary_conditions.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number>
  template <typename DofReadView, typename DofWriteView>
  void
  MeltPoolDG::CompressibleFlow::BoundaryConditions<dim, number>::
    set_conserved_variables_boundary_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::types::boundary_id                                     boundary_id,
      const DofReadView                                             &w_m,
      const DofWriteView                                            &w_p) const
  {
    if (const BoundaryType boundary_type = get_boundary_type(boundary_id);
        boundary_type == BoundaryType::slip_wall)
      {
        // homogeneous Neumann
        w_p.density()      = w_m.density();
        w_p.grad_density() = -(w_m.grad_density());

        // homogeneous Neumann
        VectorizedArrayType rho_u_dot_n = 0;
        for (unsigned int d = 0; d < dim; ++d)
          rho_u_dot_n += w_m.momentum(d) * normal[d];
        // symmetry
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p.momentum(d) = w_m.momentum(d) - 2. * rho_u_dot_n * normal[d];
            w_p.grad_momentum(d) =
              w_m.grad_momentum(d) - 2. * scalar_product(w_m.grad_momentum(d), normal) * normal;
          }
        // homogeneous Neumann
        w_p.grad_total_energy() = -(w_m.grad_total_energy());
        w_p.total_energy()      = w_m.total_energy();
      }
    else if (boundary_type == BoundaryType::no_slip_wall)
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
    else if (boundary_type == BoundaryType::inflow)
      {
        w_p.density()      = get_boundary_value(boundary_id, BoundaryType::inflow, q_point, 0);
        w_p.grad_density() = w_m.grad_density();

        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p.momentum(d) = get_boundary_value(boundary_id, BoundaryType::inflow, q_point, d + 1);
            w_p.grad_momentum(d) = w_m.grad_momentum(d);
          }

        w_p.total_energy() =
          get_boundary_value(boundary_id, BoundaryType::inflow, q_point, dim + 1);
        w_p.grad_total_energy() = w_m.grad_total_energy();
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_pressure)
      {
        // homogeneous Neumann
        w_p.density()      = w_m.density();
        w_p.grad_density() = -(w_m.grad_density());

        // Dirichlet
        auto p_dyn = dealii::VectorizedArray<number>(0.);
        for (unsigned int i = 0; i < dim; ++i)
          p_dyn += w_m.momentum(i) * w_m.momentum(i);

        p_dyn /= (w_m.density() * 2.);
        const dealii::VectorizedArray<number> pressure = get_boundary_value(
          boundary_id, BoundaryType::subsonic_outflow_fixed_pressure, q_point, 0);

        // consider equation of state for computation of inner energy from given pressure
        const dealii::VectorizedArray<number> inner_energy =
          w_m.inner_energy_from_pressure(pressure);

        w_p.total_energy()      = inner_energy + p_dyn;
        w_p.grad_total_energy() = w_m.grad_total_energy();
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_energy)
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
        w_p.total_energy() =
          get_boundary_value(boundary_id, BoundaryType::subsonic_outflow_fixed_energy, q_point, 0);
        w_p.grad_total_energy() = w_m.grad_total_energy();
      }
    else
      {
        std::cout << "ID: " << boundary_id << std::endl;
        std::cout << "Condition:" << boundary_type << std::endl;
        AssertThrow(false,
                    dealii::ExcMessage("Unknown boundary id, did "
                                       "you set a boundary condition for "
                                       "this part of the domain boundary?"));
      }
  }

  template <int dim, typename number>
  template <int n_species, typename DofReadView, typename DofWriteView>
  void
  MeltPoolDG::CompressibleFlow::BoundaryConditions<dim, number>::
    set_partial_density_boundary_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      dealii::types::boundary_id                                 boundary_id,
      const DofReadView                                         &w_m,
      const DofWriteView                                        &w_p) const
  {
    if (const BoundaryType boundary_type = get_boundary_type(boundary_id);
        boundary_type == BoundaryType::inflow)
      {
        // For inflow boundary conditions we assume that the partial densities are given and we set
        // them as Dirichlet values.

        // Remember for compressible flow we only need to prescribe n_species - 1 partial densities.
        std::array<dealii::VectorizedArray<number>, n_species - 1> dirichlet_boundary_values;

        for (unsigned int species = 0; species < n_species - 1; ++species)
          dirichlet_boundary_values[species] =
            get_boundary_value(boundary_id,
                               BoundaryType::inflow,
                               q_point,
                               CompressibleFlow::n_conserved_variables<dim, 1> + species);


        SpeciesTransport::set_boundary_value_and_gradient(
          w_p, w_m, SpeciesTransport::BoundaryConditionType::dirichlet, dirichlet_boundary_values);
      }
    else
      {
        // For all other boundary conditions we assume homogeneous Neumann for the partial
        // densities.
        SpeciesTransport::set_boundary_value_and_gradient<n_species - 1,
                                                          dealii::VectorizedArray<number>,
                                                          DofReadView,
                                                          DofWriteView>(
          w_p, w_m, SpeciesTransport::BoundaryConditionType::neumann);
      }
  }
} // namespace MeltPoolDG::CompressibleFlow
