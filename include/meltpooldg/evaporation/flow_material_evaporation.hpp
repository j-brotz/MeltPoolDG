/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, TUM, May 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <meltpooldg/flow/flow_material_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim, typename number = double>
  class IncompressibleNewtonianFluidEvaporationMaterial : public Flow::MaterialBase<dim, number>
  {
  private:
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

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
      if (quad_idx == 0)
        {
          normal_vals.reinit(cell_idx);
          normal_vals.read_dof_values_plain(normal_vector);
          normal_vals.evaluate(EvaluationFlags::values);
        }

      normal = MeltPoolDG::VectorTools::normalize<dim>(normal_vals.get_value(quad_idx), 1e-10);
    }

    Tensor<2, dim, VectorizedArray<number>>
    get_tau() final
    {
      Tensor<2, dim, VectorizedArray<number>> result;

      // compute modified rate-of-deformation tensor
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          result[i][j] = 0.5 * (grad_u[i][j] + grad_u[j][i]);

      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          result[i][j] -= div_u * normal[i] * normal[j];

      return 2. * viscosity * result;
    }

    Tensor<2, dim, VectorizedArray<number>>
    get_d_tau_d_vel_times_vel_correction() final
    {
      return get_tau();
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
  };
} // namespace MeltPoolDG::Evaporation
