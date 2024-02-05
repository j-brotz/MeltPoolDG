/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, TUM, May 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/base/tensor_accessors.h>

#include <meltpooldg/flow/incompressible_flow_material_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**
   * Class for the shear stress computation of an incompressible vapor/liquid mixture,
   * where the velocity field is not divergence-free in the interfacial region. The
   * intention is to correct the rate-of-deformation tensor to be nearly deviatoric.
   *
   * It is assumed that the stress computes as follows
   *
   *   σ = -p * I + 2 * μ * (D - tr(D) * n⊗ n)
   *
   * with the Cauchy stress tensor σ, the pressure p, the second-order identity tensor I,
   * the dynamic viscosity μ, the rate-of-deformation tensor
   *
   *   D = 0.5 * (∇u + (∇u)^T),
   *
   * the interfacial unit normal vector n and the dyadic product ⊗ .
   */
  template <int dim, typename number = double>
  class IncompressibleNewtonianFluidEvaporationMaterial
    : public Flow::IncompressibleMaterialBase<dim, number>
  {
  private:
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<number>;

  public:
    IncompressibleNewtonianFluidEvaporationMaterial(
      const ScratchData<dim>                                                    &scratch_data,
      const std::function<const VectorizedArray<number> &(const unsigned int cell,
                                                          const unsigned int q)> get_viscosity,
      const BlockVectorType                                                     &normal_vector,
      const VectorType                                                          &heaviside,
      const unsigned int                                                         normal_dof_idx,
      const unsigned int ls_hanging_nodes_dof_idx,
      const unsigned int velocity_quad_idx)
      : scratch_data(scratch_data)
      , get_viscosity(get_viscosity)
      , normal_vector(normal_vector)
      , heaviside(heaviside)
      , normal_dof_idx(normal_dof_idx)
      , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx)
      , velocity_quad_idx(velocity_quad_idx)
      , normal_vals(scratch_data.get_matrix_free(), normal_dof_idx, velocity_quad_idx)
      , ls_vals(scratch_data.get_matrix_free(), ls_hanging_nodes_dof_idx, velocity_quad_idx)
    {}

    /**
     * Reinit function to be used to compute quadrature point variables (grad_u, div_u, viscosity,
     * normal). The gradient of the velocity vector @p velocity_gradient, the current cell index
     * @p cell_idx and the current quadrature point index @p quad_idx are known.
     */
    void
    reinit(const Tensor<2, dim, VectorizedArray<number>> &velocity_gradient,
           const unsigned int                             cell_idx,
           const unsigned int                             quad_idx) final
    {
      grad_u = velocity_gradient;
      div_u  = trace(velocity_gradient);

      viscosity = get_viscosity(cell_idx, quad_idx);

      // read normal vector values only for first quadrature point
      if (quad_idx == 0 || cell_idx != cell)
        {
          normal_vals.reinit(cell_idx);
          normal_vals.read_dof_values_plain(normal_vector);
          normal_vals.evaluate(EvaluationFlags::values);

          ls_vals.reinit(cell_idx);
          ls_vals.read_dof_values_plain(heaviside);
          ls_vals.evaluate(EvaluationFlags::values);
        }

      normal = MeltPoolDG::VectorTools::normalize<dim>(normal_vals.get_value(quad_idx), 1e-10);
      hs     = ls_vals.get_value(quad_idx);

      cell = cell_idx;
    }

    /**
     * Return the deviatoric stress τ for the given velocity gradient ∇u, set by reinit()
     *
     *   τ(∇u) = 2 * μ * (D - tr(D) * n⊗ n)
     *
     * with the dynamic viscosity μ, the rate-of-deformation tensor
     *
     *   D = 0.5 * (∇u + (∇u)^T)
     *
     * the interfacial unit normal vector n and the dyadic product ⊗ .
     */
    Tensor<2, dim, VectorizedArray<number>>
    get_tau() final
    {
      const auto mask = compare_and_apply_mask<SIMDComparison::less_than>(
        hs, 1.0, compare_and_apply_mask<SIMDComparison::greater_than>(hs, 0, 1.0, 0.0), 0);

      return 2. * viscosity *
             (0.5 * (grad_u + transpose(grad_u)) - mask * div_u * outer_product(normal, normal));
    }

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
     * with the shape functions N and the material tangent written in index notation
     *
     *   ∂ τ
     *  ----- = C     = 2 * μ * (δ     - δ   * n  n  )
     *   ∂ ∇u    ijkl             ijkl    ij    k   l
     *
     * with the Kronecker delta δ.
     *
     */
    Tensor<2, dim, VectorizedArray<number>>
    get_vmult_d_tau_d_grad_vel() final
    {
      // Since the velocity DoFs occur linear in the expression for tau, the vmult of the derivative
      // is the same operation, except with a different meaning of grad_u, i.e. shape functions
      // gradient times correction values for the velocity DoFs from the linear solver.
      return get_tau();

      // @note alternative, in case we want to convert this function to a real material tangent and
      // leave the multiplication with the velocity gradient to the external code.
      //
      // const auto identity  = Tensor<4,dim,VectorizedArray<number>>(identity_tensor<dim,
      // VectorizedArray<number>>()); const auto identity2 =
      // Tensor<2,dim,VectorizedArray<number>>(unit_symmetric_tensor<dim,
      // VectorizedArray<number>>());

      // const auto temp = 2. * viscosity * (identity - outer_product(identity2,
      // outer_product(normal, normal))); return double_contract<0,0,1,1>(temp, grad_u);
    }

    /**
     * Update ghost values of the vectors, to be used to compute the deviatoric stress.
     */
    void
    update_ghost_values() final
    {
      normal_update_ghosts = !normal_vector.has_ghost_elements();
      if (normal_update_ghosts)
        normal_vector.update_ghost_values();

      ls_update_ghosts = !heaviside.has_ghost_elements();
      if (ls_update_ghosts)
        heaviside.update_ghost_values();
    }

    /**
     * Zero out ghost values of the vectors, to be used to compute the deviatoric stress.
     */
    void
    zero_out_ghost_values() final
    {
      if (normal_update_ghosts)
        normal_vector.zero_out_ghost_values();

      if (ls_update_ghosts)
        heaviside.zero_out_ghost_values();
    }

  private:
    const ScratchData<dim> &scratch_data;
    const std::function<const VectorizedArray<number> &(const unsigned int cell,
                                                        const unsigned int q)>
                                       get_viscosity;
    const BlockVectorType             &normal_vector;
    const VectorType                  &heaviside;
    const unsigned int                 normal_dof_idx;
    const unsigned int                 ls_hanging_nodes_dof_idx;
    const unsigned int                 velocity_quad_idx;
    FECellIntegrator<dim, dim, number> normal_vals;
    FECellIntegrator<dim, 1, number>   ls_vals;

    // temporary quadrature point values
    Tensor<2, dim, VectorizedArray<number>> grad_u;
    VectorizedArray<number>                 div_u;
    VectorizedArray<number>                 viscosity;
    Tensor<1, dim, VectorizedArray<number>> normal;
    VectorizedArray<number>                 hs;
    unsigned int                            cell;

    mutable bool ls_update_ghosts     = true;
    mutable bool normal_update_ghosts = true;
  };
} // namespace MeltPoolDG::Evaporation
