#pragma once

namespace MeltPoolDG::Flow
{
  /**
   * @brief Base class for the shear stress computation of an incompressible fluid material
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
  template <int dim, typename number>
  class IncompressibleMaterialBase
  {
  public:
    virtual ~IncompressibleMaterialBase() = default;

    /**
     * @brief Reinit function to be used to compute quadrature point variables.
     *
     * @param velocity_gradient Gradient of the velocity vector.
     * @param cell_idx Current cell index.
     * @param quad_idx Current quadrature point index.
     */
    virtual void
    reinit(const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &velocity_gradient,
           const unsigned int                                             cell_idx,
           const unsigned int                                             quad_idx) = 0;

    /**
     * @brief Compute the deviatoric stress τ for the given velocity gradient ∇u, set by reinit().
     *
     * @return Deviatoric stress tensor τ(∇u).
     */
    virtual dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    get_tau() = 0;

    /**
     * @brief Compute the material tangent of the deviatoric stress and perform a vmult with the
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
     * @return Material tangent of the deviatoric stress tensor multiplied with the gradient of the
     * nodal DoF values δu.
     */
    virtual dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    get_vmult_d_tau_d_grad_vel() = 0;

    /**
     * @brief Update ghost values of the vectors, to be used to compute the deviatoric stress.
     *
     * Default: do nothing
     */
    virtual void
    update_ghost_values()
    {
      // do nothing
    }

    /**
     * @brief Zero out ghost values of the vectors, to be used to compute the deviatoric stress.
     *
     * Default: do nothing
     */
    virtual void
    zero_out_ghost_values()
    {
      // do nothing
    }
  };

} // namespace MeltPoolDG::Flow
