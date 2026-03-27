#pragma once

#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * Calculate the convective flux F_c for the compressible Navier-Stokes equations.
   *
   * @param conserved_variables View on the local values of the conserved variables, providing
   * convenient accessor functions for the quantities needed to compute the convective flux.
   *
   * @tparam value_type Type of the conserved variables.
   */
  template <int dim, typename number, CompressibleFlow::IsConservedStateCompatible<dim> value_type>
  inline DEAL_II_ALWAYS_INLINE //
    CompressibleFlow::FluxType<dim, number>
    convective_flux(
      const CompressibleFlow::DofStateView<dim, number, const value_type> &conserved_variables)
  {
    using FluxType = CompressibleFlow::FluxType<dim, number>;

    FluxType                                  flux;
    CompressibleFlow::FluxView<dim, FluxType> flux_view(flux);
    const dealii::VectorizedArray<number>     pressure = conserved_variables.pressure();

    for (unsigned int d = 0; d < dim; ++d)
      {
        flux_view.density_flux()[d] = conserved_variables.momentum(d);
        for (unsigned int e = 0; e < dim; ++e)
          flux_view.momentum_flux(e)[d] =
            conserved_variables.momentum(e) * conserved_variables.velocity(d);
        flux_view.momentum_flux(d)[d] += pressure;
        flux_view.energy_flux()[d] =
          conserved_variables.velocity(d) * (conserved_variables.total_energy() + pressure);
      }
    return flux;
  }

  /**
   * Calculate the viscous stress tensor for the compressible Navier-Stokes equations.
   *
   * @param grad_velocity Gradient of the velocity field.
   * @param dynamic_viscosity Dynamic viscosity of the fluid.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
    viscous_stress_tensor(
      const dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
                                           &grad_velocity,
      const dealii::VectorizedArray<number> dynamic_viscosity)
  {
    const dealii::VectorizedArray<number> div_u = 2. / 3. * trace(grad_velocity);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> out;
    for (unsigned int d = 0; d < dim; ++d)
      {
        for (unsigned int e = 0; e < dim; ++e)
          out[d][e] = dynamic_viscosity * (grad_velocity[d][e] + grad_velocity[e][d]);
        out[d][d] -= dynamic_viscosity * div_u;
      }

    return out;
  }

  /**
   * Calculate the diffusive (viscous) flux F_d for the compressible Navier-Stokes equations.
   *
   * @param conserved_variables A view on the local values of the conserved variables and their
   * gradients, providing convenient accessor functions for the quantities needed to compute the
   * diffusive flux, such as velocity, velocity gradients, temperature gradients, dynamic viscosity,
   * and thermal conductivity.
   *
   * @tparam value_type Type of the conserved variables.
   * @tparam gradient_type Type of the gradients of the conserved variables.
   */
  template <int dim,
            typename number,
            CompressibleFlow::IsConservedStateCompatible<dim>    value_type,
            CompressibleFlow::IsConservedGradientCompatible<dim> gradient_type>
  inline DEAL_II_ALWAYS_INLINE //
    CompressibleFlow::FluxType<dim, number>
    diffusive_flux(
      const CompressibleFlow::
        DofValueAndGradientStateView<dim, number, const value_type, const gradient_type>
          &conserved_variables)
  {
    using FluxType = CompressibleFlow::FluxType<dim, number>;

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
      conserved_variables.velocity();

    const dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
      viscous_stress = viscous_stress_tensor(conserved_variables.grad_velocity(),
                                             dealii::VectorizedArray<number>(
                                               conserved_variables.dynamic_viscosity()));

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> neg_heat_flux =
      conserved_variables.thermal_conductivity() * conserved_variables.grad_temperature();

    FluxType                                  flux;
    CompressibleFlow::FluxView<dim, FluxType> flux_view(flux);
    for (unsigned int d = 0; d < dim; ++d)
      {
        // density
        flux_view.density_flux()[d] = 0.0;

        // momentum
        for (unsigned int e = 0; e < dim; ++e)
          flux_view.momentum_flux(e)[d] = viscous_stress[e][d];

        // energy
        flux_view.energy_flux()[d] = neg_heat_flux[d];
        for (unsigned int e = 0; e < dim; ++e)
          flux_view.energy_flux()[d] += velocity[e] * viscous_stress[d][e];
      }

    return flux;
  }


  template <int dim, typename number>
  struct CompressibleConvectiveFlux
  {
    using ValueType = CompressibleFlow::ConservedVariablesType<dim, number>;
    using FluxType  = CompressibleFlow::FluxType<dim, number>;

    using DofStateViewType = CompressibleFlow::DofStateView<dim, number, const ValueType>;

    explicit CompressibleConvectiveFlux(
      const Flow::CompressibleFluidMaterialPhaseData<number> &material)
      : material(material)
    {}

    /**
     * Calculate the convective flux F_c for the compressible Navier-Stokes equations.
     *
     * @param conserved_variables Local values of the conserved variables.
     */
    FluxType
    flux(const ValueType &conserved_variables) const
    {
      DofStateViewType conserved_variables_view(conserved_variables, material);
      return convective_flux<dim, number>(conserved_variables_view);
    }

    /**
     * Calculate the local maximum wave speed (eigenvalue) for the Lax-Friedrichs numerical flux.
     *
     * @param u_m Local values of the conserved variables on the inner face.
     * @param u_p Local values of the conserved variables on the outer face.
     */
    dealii::VectorizedArray<number>
    lambda(const ValueType &u_m, const ValueType &u_p) const
    {
      DofStateViewType u_m_view(u_m, material);
      DofStateViewType u_p_view(u_p, material);

      const auto velocity_m = u_m_view.velocity();
      const auto velocity_p = u_p_view.velocity();

      const auto sound_speed_p = u_p_view.speed_of_sound();
      const auto sound_speed_m = u_m_view.speed_of_sound();

      const auto sound_speed_p2 = sound_speed_p * sound_speed_p;
      const auto sound_speed_m2 = sound_speed_m * sound_speed_m;

      const auto lambda = 0.5 * std::sqrt(std::max(velocity_p.norm_square() + sound_speed_p2,
                                                   velocity_m.norm_square() + sound_speed_m2));
      return lambda;
    }

  private:
    const Flow::CompressibleFluidMaterialPhaseData<number> &material;
  };


  template <int dim, typename number>
  struct CompressibleDiffusiveFlux
  {
    using ValueType    = CompressibleFlow::ConservedVariablesType<dim, number>;
    using GradientType = CompressibleFlow::ConservedVariablesGradientType<dim, number>;
    using FluxType     = CompressibleFlow::FluxType<dim, number>;

    using DofValueAndGradientStateViewType = CompressibleFlow::
      DofValueAndGradientStateView<dim, number, const ValueType, const GradientType>;

    /**
     * Constructor, storing references to the EOS utilities and material data needed for flux
     * calculations.
     */
    explicit CompressibleDiffusiveFlux(
      const Flow::CompressibleFluidMaterialPhaseData<number> &material)
      : material(material)
    {}

    /**
     * Calculate the diffusive flux F_d for the compressible Navier-Stokes equations.
     *
     * @param u Local values of the conserved variables.
     * @param grad_u Local gradients of the conserved variables.
     */
    FluxType
    flux(const ValueType &u, const GradientType &grad_u) const
    {
      DofValueAndGradientStateViewType conserved_variables_view(u, grad_u, material);
      return diffusive_flux<dim, number>(conserved_variables_view);
    }

  private:
    const Flow::CompressibleFluidMaterialPhaseData<number> &material;
  };
} // namespace MeltPoolDG::Flow