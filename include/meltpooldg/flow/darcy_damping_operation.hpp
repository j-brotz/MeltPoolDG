#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/flow/darcy_damping_data.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>


namespace MeltPoolDG::Flow
{
  using namespace dealii;

  /**
   * This class computes the Darcy damping force (or Darcy source term).
   *
   *         /              \
   *  f_d =  | N_a, K N_b u |
   *         \              /
   *                         Ω
   *
   * with the isotropic permeability of the mushy zone K given by the Kozeny-Carman equation:
   *
   *                 fs²
   *  K = - C ----------------
   *           ( 1- fs )³ + b
   *
   * with the solid fraction fs \in [0, 1], the morphology of the mushy zone C and the parameter b
   * to avoid division by zero.
   *
   * @note The constant b must be greater than zero.
   *
   * Voller, V. R., & Prakash, C. (1987). A fixed grid numerical modelling methodology for
   * convection-diffusion mushy region phase-change problems. International Journal of Heat and Mass
   * Transfer, 30(8), 1709–1719. https://doi.org/10.1016/0017-9310(87)90317-6
   */
  template <int dim>
  class DarcyDampingOperation
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const double                                   mushy_zone_morphology;
    const double                                   avoid_div_zero_constant;
    const ScratchData<dim>                        &scratch_data;
    const unsigned int                             flow_vel_hanging_nodes_dof_idx;
    const unsigned int                             flow_quad_idx;
    const unsigned int                             solid_dof_idx;
    mutable VectorType                             damping;
    mutable AlignedVector<VectorizedArray<double>> damping_at_q;

  public:
    DarcyDampingOperation(const DarcyDampingData<double> &data_in,
                          const ScratchData<dim>         &scratch_data,
                          const unsigned int              flow_vel_hanging_nodes_dof_idx,
                          const unsigned int              flow_quad_idx,
                          const unsigned int              solid_dof_idx);

    /**
     * Compute the contribution of the Darcy damping force into a force vector @param force_rhs.
     *
     * @note The Darcy damping coefficient is computed based on the @param solid_fraction_vec.
     */
    void
    compute_darcy_damping(VectorType       &force_rhs,
                          const VectorType &velocity_vec,
                          const VectorType &solid_fraction_vec,
                          const bool        zero_out = true);

    /**
     * Loop over all cells and quadrature points (defined via @p flow_quad_idx),
     * compute the Darcy damping coefficient and store it.
     *
     * @param material Material object holding the phase parameters.
     * @param ls_as_heaviside Indicator function between gas and heavy phase.
     * @param temperature DoF vector of the temperature field.
     * @param ls_hanging_nodes_dof_idx DoF index of the indicator field inside ScratchData.
     * @param temp_dof_idx DoF index of the temperature field.
     */
    void
    set_darcy_damping_at_q(const Material<double> &material,
                           const VectorType       &ls_as_heaviside,
                           const VectorType       &temperature,
                           const unsigned int      ls_hanging_nodes_dof_idx,
                           const unsigned int      temp_dof_idx);

    /**
     * Compute the contribution of the Darcy damping force into a force vector @param force_rhs.
     *
     * @note To use this function, the Darcy damping coefficient at the quadrature points of every
     * cell, stored in @param damping_at_q, must be set IN ADVANCE. The latter is accessible cellwise
     * by get_damping(cell, q) or as a global vector by get_damping_at_q().
     */
    void
    compute_darcy_damping(VectorType       &force_rhs,
                          const VectorType &velocity_vec,
                          const bool        zero_out = true);

    void
    reinit();

    /**
     * Compute the Darcy damping coefficient based on a given @param solid_fraction.
     */
    VectorizedArray<double>
    compute_darcy_damping_coefficient(const VectorizedArray<double> &solid_fraction) const;

    /**
     * Store the damping coefficients in a global DoF vector and attach it to the output data.
     */
    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    /**
     * Getter functions for the damping coefficients cellwise at each quadrature point.
     */
    VectorizedArray<double> &
    get_damping(const unsigned int cell, const unsigned int q);

    const VectorizedArray<double> &
    get_damping(const unsigned int cell, const unsigned int q) const;

    /**
     * Getter function for the vector of damping coefficients, holding the values at each cell and
     * at each quadrature point.
     */
    AlignedVector<VectorizedArray<double>> &
    get_damping_at_q();
  };
} // namespace MeltPoolDG::Flow
