/**
 * @file riemann_solvers.hpp
 * @brief This file contains the implementation of different Riemann solvers for hyperbolic systems
 * of conservation laws. The Riemann solvers are used to compute the numerical fluxes at the
 * interfaces between cells in a finite volume or discontinuous Galerkin method.
 */

#pragma once

#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/dealii_tensor.hpp>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  lax_friedrichs_flux(const dealii::Tensor<1, n_components, number>                         &u_m,
                      const dealii::Tensor<1, n_components, number>                         &u_p,
                      const dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>> &flux_m,
                      const dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>> &flux_p,
                      const dealii::Tensor<1, dim, number> &unit_normal,
                      const number                          maximum_wave_speed)
  {
    return contract_average_tensor_with_vector(flux_m, flux_p, unit_normal) +
           0.5 * maximum_wave_speed * (u_m - u_p);
  }

} // namespace MeltPoolDG
