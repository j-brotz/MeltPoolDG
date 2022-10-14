/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, TUM, May 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  /**
   * Base class for the shear stress computation of an incompressible fluid material
   *
   * It is assumed that the stress computation has the form
   *
   *   σ = -p * I + τ
   *
   * with the Cauchy stress tensor σ, the pressure p, the second-order Identity tensor I
   * and the deviatoric stress tensor τ.
   *
   * @note The pressure p is given from the incompressibility constraint from the flow
   * solver and solely the deviatoric stress τ will be computed herein. Thus, the material
   * law relates solely the deviatoric (shear) stress tensor to the velocity gradient ∇u:
   *
   *   τ(∇u)
   *
   */
  template <int dim, typename number = double>
  class IncompressibleMaterialBase
  {
  public:
    /**
     * Reinit function to be used to compute quadrature point variables. The gradient of
     * the velocity vector @p velocity_gradient, the current cell index @p cell_idx and
     * the current quadrature point index @p quad_idx are known.
     */
    virtual void
    reinit(const Tensor<2, dim, VectorizedArray<number>> &velocity_gradient,
           const unsigned int                             cell_idx,
           const unsigned int                             quad_idx) = 0;

    /**
     * Return the deviatoric stress τ for the given velocity gradient ∇u, set by reinit().
     *  τ(∇u)
     */
    virtual Tensor<2, dim, VectorizedArray<number>>
    get_tau() = 0;

    /**
     * Return the material tangent of the deviatoric stress and perform a vmult with the
     * gradient of the nodal DoF values δu, set by reinit(),
     *
     *    ∂ τ
     *   ------ * ∇N * δu
     *    ∂ ∇u
     * |________|
     *  material
     *  tangent
     *
     * with the shape functions N.
     *
     */
    virtual Tensor<2, dim, VectorizedArray<number>>
    get_vmult_d_tau_d_grad_vel() = 0;

    /**
     * Update ghost values of the vectors, to be used to compute the deviatoric stress.
     * Default: do nothing
     */
    virtual void
    update_ghost_values()
    {
      // do nothing
    }

    /**
     * Zero out ghost values of the vectors, to be used to compute the deviatoric stress.
     * Default: do nothing
     */
    virtual void
    zero_out_ghost_values()
    {
      // do nothing
    }
  };

} // namespace MeltPoolDG::Flow
