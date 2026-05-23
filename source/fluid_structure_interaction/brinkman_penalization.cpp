#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/particles/particle_accessor.h>

#include "meltpooldg/particles/dem_util.hpp"
#include "meltpooldg/utilities/matrix_free_util.hpp"
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <utility>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::BrinkmanObstacleForce<dim, number, ObstacleType>::BrinkmanObstacleForce(
  const ObstacleField<dim, number, ObstacleType> &obstacle_field,
  const VectorType                               &solution,
  const MatrixFreeContext<dim, number>           &matrix_free,
  const BrinkmanPenalizationData<number>         &data)
  : brinkman_penalization_data(data)
  , cell_obstacle_cache(obstacle_field)
  , matrix_free(matrix_free)
  , solution(solution)
{}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::BrinkmanObstacleForce<dim, number, ObstacleType>::add_load_to_obstacles(
  ObstacleField<dim, number, ObstacleType> &obstacle_field) const
{
  std::function<void(const dealii::MatrixFree<dim, number> &,
                     dealii::Tensor<1, dim, number> &,
                     const VectorType &,
                     const std::pair<unsigned, unsigned> &)>
    local_apply_cell = [&](const dealii::MatrixFree<dim, number> &,
                           dealii::Tensor<1, dim, number> &,
                           const VectorType                    &solution,
                           const std::pair<unsigned, unsigned> &cell_range) {
      FECellIntegrator<dim, dim + 2, number> phi(matrix_free.mf,
                                                 matrix_free.dof_idx,
                                                 matrix_free.quad_idx);

      for (unsigned cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          std::vector<DEMParticleAccessor<dim, number>> relevant_obstacle =
            obstacle_field.get_obstacles_in_cell(
              MeltPoolDG::cells_in_cell_batch(matrix_free.mf, cell));
          if (not relevant_obstacle.empty())
            {
              phi.gather_evaluate(solution, dealii::EvaluationFlags::values);

              for (const unsigned int q : phi.quadrature_point_indices())
                {
                  const auto                                              w_q = phi.get_value(q);
                  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_momentum;
                  for (int d = 0; d < dim; ++d)
                    fluid_momentum[d] = w_q[d + 1];

                  auto add_force_and_torque_to_particle = [&](DEMParticleAccessor<dim, number>
                                                                &obstacle) {
                    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>            force;
                    dealii::Tensor<1, axial_dim<dim>, dealii::VectorizedArray<number>> torque;

                    const dealii::VectorizedArray<number> mask =
                      mask_function<dim, number, dealii::VectorizedArray<number>, ObstacleType>(
                        brinkman_penalization_data.mask_function_type,
                        phi.quadrature_point(q),
                        obstacle);
                    force = mask / brinkman_penalization_data.permeability *
                            (fluid_momentum -
                             w_q[0] * obstacle.get_local_velocity(phi.quadrature_point(q))) *
                            phi.JxW(q);

                    if constexpr (dim == 2)
                      {
                        auto vector_to_center_of_gravity =
                          obstacle
                            .template vector_to_center_of_gravity<dealii::VectorizedArray<number>>(
                              phi.quadrature_point(q));

                        torque[0] = -vector_to_center_of_gravity[0] * force[1] +
                                    vector_to_center_of_gravity[1] * force[0];
                      }
                    if constexpr (dim == 3)
                      {
                        torque = -dealii::cross_product_3d(
                          obstacle
                            .template vector_to_center_of_gravity<dealii::VectorizedArray<number>>(
                              phi.quadrature_point(q)),
                          force);
                      }

                    dealii::Tensor<1, dim, number>            summed_force;
                    dealii::Tensor<1, axial_dim<dim>, number> summed_torque;

                    for (int i = 0; i < dim; ++i)
                      summed_force[i] = force[i].sum();
                    for (unsigned i = 0; i < axial_dim<dim>; ++i)
                      summed_torque[i] = torque[i].sum();

                    obstacle.add_force(summed_force);
                    obstacle.add_torque(summed_torque);
                  };

                  for (DEMParticleAccessor<dim, number> &obstacle : relevant_obstacle)
                    add_force_and_torque_to_particle(obstacle);
                }
            }
        }
    };

  dealii::Tensor<1, dim, number> particle_force_dummy;
  matrix_free.mf.cell_loop(local_apply_cell, particle_force_dummy, solution);

  obstacle_field.compress();
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>::
  BrinkmanPenalizationResidualContribution(
    const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
    const BrinkmanPenalizationData<number>         &brinkman_penalization_data)
  : brinkman_penalization_data(brinkman_penalization_data)
  , cell_obstacle_cache(obstacle_handler)
{}

template <int dim, typename number, typename ObstacleType>
auto
MeltPoolDG::BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>::value(
  const number,
  const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cell_iterators,
  const dealii::Point<dim, dealii::VectorizedArray<number>>          &q_point,
  const ConservedVariablesType                                       &w_q) -> ConservedVariablesType
{
  cell_obstacle_cache.update_cache(cell_iterators);

  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_momentum;
  for (int d = 0; d < dim; ++d)
    fluid_momentum[d] = w_q[d + 1];

  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum_penalty;
  dealii::VectorizedArray<number>                         energy_penalty = 0.;
  for (const DEMParticleAccessor<dim, number> &obstacle : cell_obstacle_cache.obstacle_cache)
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> local_obstacle_velocity =
        obstacle.get_local_velocity(q_point);
      auto mask = mask_function<dim, number, dealii::VectorizedArray<number>, ObstacleType>(
        brinkman_penalization_data.mask_function_type, q_point, obstacle);
      momentum_penalty -= mask / brinkman_penalization_data.permeability *
                          (fluid_momentum - w_q[0] * local_obstacle_velocity);
      energy_penalty += mask / brinkman_penalization_data.permeability *
                        (fluid_momentum - w_q[0] * local_obstacle_velocity) * fluid_momentum /
                        w_q[0];
    }

  ConservedVariablesType fluid_force;
  for (int d = 0; d < dim; ++d)
    fluid_force[d + 1] = momentum_penalty[d];
  fluid_force[dim + 1] = energy_penalty;
  return fluid_force;
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>::
  BrinkmanPenalizationJacobianContribution(
    const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
    const BrinkmanPenalizationData<number>         &brinkman_penalization_data)
  : brinkman_penalization_data(brinkman_penalization_data)
  , cell_obstacle_cache(obstacle_handler)
{}

template <int dim, typename number, typename ObstacleType>
auto
MeltPoolDG::BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>::value(
  const number,
  const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cell_iterators,
  const dealii::Point<dim, dealii::VectorizedArray<number>>          &q_point,
  const ConservedVariablesType                                       &w_q,
  const ConservedVariablesType &delta_w_q) -> ConservedVariablesType
{
  cell_obstacle_cache.update_cache(cell_iterators);

  if (cell_obstacle_cache.obstacle_cache.empty())
    return ConservedVariablesType();

  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_momentum;
  for (int d = 0; d < dim; ++d)
    fluid_momentum[d] = w_q[d + 1];

  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_momentum_differential_change;
  for (int d = 0; d < dim; ++d)
    fluid_momentum_differential_change[d] = delta_w_q[d + 1];

  dealii::VectorizedArray<number>                         energy_penalty_differential_change(0.);
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum_penalty_differential_change;
  for (const DEMParticleAccessor<dim, number> &obstacle : cell_obstacle_cache.obstacle_cache)
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> local_obstacle_velocity =
        obstacle.get_local_velocity(q_point);

      auto mask = mask_function<dim, number, dealii::VectorizedArray<number>, ObstacleType>(
        brinkman_penalization_data.mask_function_type, q_point, obstacle);

      energy_penalty_differential_change +=
        mask / brinkman_penalization_data.permeability *
        (dealii::scalar_product(fluid_momentum, fluid_momentum) / (w_q[0] * w_q[0]) * delta_w_q[0] -
         2. / w_q[0] * dealii::scalar_product(fluid_momentum, fluid_momentum_differential_change) +
         dealii::scalar_product(local_obstacle_velocity, fluid_momentum_differential_change));

      for (int i = 0; i < dim; ++i)
        momentum_penalty_differential_change[i] -=
          mask / brinkman_penalization_data.permeability *
          (fluid_momentum_differential_change[i] - local_obstacle_velocity[i] * delta_w_q[0]);
    }
  ConservedVariablesType penalty_differential_change;
  penalty_differential_change[0] = 0.;
  for (int d = 0; d < dim; ++d)
    penalty_differential_change[d + 1] = momentum_penalty_differential_change[d];
  penalty_differential_change[dim + 1] = energy_penalty_differential_change;
  return penalty_differential_change;
}

template struct MeltPoolDG::
  BrinkmanPenalizationResidualContribution<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  BrinkmanPenalizationResidualContribution<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  BrinkmanPenalizationResidualContribution<3, double, MeltPoolDG::SphericalParticle<3, double>>;
template struct MeltPoolDG::
  BrinkmanPenalizationJacobianContribution<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  BrinkmanPenalizationJacobianContribution<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  BrinkmanPenalizationJacobianContribution<3, double, MeltPoolDG::SphericalParticle<3, double>>;


template struct MeltPoolDG::
  BrinkmanObstacleForce<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  BrinkmanObstacleForce<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  BrinkmanObstacleForce<3, double, MeltPoolDG::SphericalParticle<3, double>>;
