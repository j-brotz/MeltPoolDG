#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <meltpooldg/core/material.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/darcy_damping_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>


namespace MeltPoolDG::Flow
{
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
  template <int dim, typename number>
  class DarcyDampingOperation
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    const number                                                   mushy_zone_morphology;
    const number                                                   avoid_div_zero_constant;
    const ScratchData<dim, dim, number>                           &scratch_data;
    const unsigned int                                             flow_vel_hanging_nodes_dof_idx;
    const unsigned int                                             flow_quad_idx;
    mutable VectorType                                             damping;
    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> damping_at_q;

  public:
    DarcyDampingOperation(const DarcyDampingData<number>      &data_in,
                          const ScratchData<dim, dim, number> &scratch_data,
                          const unsigned int                   flow_vel_hanging_nodes_dof_idx,
                          const unsigned int                   flow_quad_idx);

    void
    reinit();

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
    set_darcy_damping_at_q(const Material<number> &material,
                           const VectorType       &ls_as_heaviside,
                           const VectorType       &temperature,
                           const unsigned int      ls_hanging_nodes_dof_idx,
                           const unsigned int      temp_dof_idx);

    /**
     * Assemble the Darcy damping force contribution into the right-hand side (RHS) vector @p force_rhs.
     *
     * This function computes and adds (or sets, if @p zero_out is true) the contribution of the Darcy damping
     * force, based on the provided velocity vector @p velocity_vec.
     *
     * @param force_rhs   The global force vector to which the damping contribution will be added.
     * @param velocity_vec The velocity field used in evaluating the damping term.
     * @param zero_out    If true, the @p force_rhs vector is cleared before the contribution is added.
     *
     * @note The damping coefficient must be precomputed at all quadrature points of each cell and made
     * accessible via get_damping(cell, q) or get_damping_at_q(). This setup must be completed
     * *before* calling this function.
     */
    void
    assemble_rhs(VectorType &force_rhs, const VectorType &velocity_vec, const bool zero_out = true);

    /**
     * Store the damping coefficients in a global DoF vector and attach it to the output data.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /**
     * Getter functions for the damping coefficients cellwise at each quadrature point.
     */
    dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q);

    const dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q) const;

  private:
    mutable dealii::Vector<number> damping_output;

    /**
     * Compute the Darcy damping coefficient based on a given @param solid_fraction.
     */
    dealii::VectorizedArray<number>
    compute_darcy_damping_coefficient(const dealii::VectorizedArray<number> &solid_fraction) const;

    /**
     * Getter function for the vector of damping coefficients, holding the values at each cell and
     * at each quadrature point.
     */
    dealii::AlignedVector<dealii::VectorizedArray<number>> &
    get_damping_at_q();
  };
} // namespace MeltPoolDG::Flow
