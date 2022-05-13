/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, TUM, May 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/base/tensor_accessors.h>

#include <meltpooldg/flow/flow_material_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>
namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim, typename number = double>
  class IncompressibleNewtonianFluidEvaporationMaterial
    : public Flow::IncompressibleMaterialBase<dim, number>
  {
  private:
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<number>;

  public:
    IncompressibleNewtonianFluidEvaporationMaterial(
      const ScratchData<dim> &                                                   scratch_data,
      const std::function<const VectorizedArray<number> &(const unsigned int cell,
                                                          const unsigned int q)> get_viscosity,
      const BlockVectorType &                                                    normal_vector,
      const unsigned int                                                         normal_dof_idx,
      const unsigned int                                                         velocity_quad_idx)
      : scratch_data(scratch_data)
      , get_viscosity(get_viscosity)
      , normal_vector(normal_vector)
      , normal_dof_idx(normal_dof_idx)
      , velocity_quad_idx(velocity_quad_idx)
      , normal_vals(scratch_data.get_matrix_free(), normal_dof_idx, velocity_quad_idx)
    {}

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
        }

      normal = MeltPoolDG::VectorTools::normalize<dim>(normal_vals.get_value(quad_idx), 1e-10);

      cell = cell_idx;
    }

    Tensor<2, dim, VectorizedArray<number>>
    get_tau() final
    {
      return 2. * viscosity *
             (0.5 * (grad_u + transpose(grad_u)) - div_u * outer_product(normal, normal));
    }

    Tensor<2, dim, VectorizedArray<number>>
    get_vmult_d_tau_d_grad_vel() final
    {
      // Since the velocity DoFs occur linear in the expression for tau, the vmult of the derivative
      // is the same operation, except with a different meaning of grad_u, i.e. shape functions
      // gradient times correction values for the velocity DoFs from the linear solver.
      return get_tau();

      // @note: alternative, in case we want to convert this function to a real material tangent and
      // leave the multiplication with the velocity gradient to the external code.
      //
      // const auto identity  = Tensor<4,dim,VectorizedArray<number>>(identity_tensor<dim,
      // VectorizedArray<number>>()); const auto identity2 =
      // Tensor<2,dim,VectorizedArray<number>>(unit_symmetric_tensor<dim,
      // VectorizedArray<number>>());

      // const auto temp = 2. * viscosity * (identity - outer_product(identity2,
      // outer_product(normal, normal))); return double_contract<0,0,1,1>(temp, grad_u);
    }

    void
    update_ghost_values() final
    {
      normal_vector.update_ghost_values();
    }

    void
    zero_out_ghost_values() final
    {
      normal_vector.zero_out_ghost_values();
    }

  private:
    const ScratchData<dim> &scratch_data;
    const std::function<const VectorizedArray<number> &(const unsigned int cell,
                                                        const unsigned int q)>
                                       get_viscosity;
    const BlockVectorType &            normal_vector;
    const unsigned int                 normal_dof_idx;
    const unsigned int                 velocity_quad_idx;
    FECellIntegrator<dim, dim, number> normal_vals;

    // temporary quadrature point values
    Tensor<2, dim, VectorizedArray<number>> grad_u;
    VectorizedArray<number>                 div_u;
    VectorizedArray<number>                 viscosity;
    Tensor<1, dim, VectorizedArray<number>> normal;
    unsigned int                            cell;
  };
} // namespace MeltPoolDG::Evaporation
