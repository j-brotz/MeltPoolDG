#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_obstacle_to_fluid.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <utility>

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>::
  BrinkmanPenalizationResidualContribution(
    const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
    const FluidStructureInteractionData<number>    &brinkman_penalization_data,
    typename BrinkmanPenalizationCellScratchData<dim, number, ObstacleType>::MaskFunctionType
      mask_function)
  : brinkman_cell_scratch_data(obstacle_handler,
                               brinkman_penalization_data,
                               std::move(mask_function))
{}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>::cell_operation(
  const dealii::MatrixFree<dim, number> &matrix_free,
  const unsigned int                     cell_batch_id,
  const unsigned int /*dof_idx*/)
{
  find_relevant_obstacles_in_cell_batch<dim, number, ObstacleType>(brinkman_cell_scratch_data,
                                                                   matrix_free,
                                                                   cell_batch_id);
}

template <int dim, typename number, typename ObstacleType>
auto
MeltPoolDG::BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>::quad_operation(
  const number,
  const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
  const ConservedVariablesType                              &w_q) -> ConservedVariablesType
{
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_momentum;
  for (int d = 0; d < dim; ++d)
    fluid_momentum[d] = w_q[d + 1];

  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum_penalty;
  dealii::VectorizedArray<number>                         energy_penalty = 0.;
  for (auto obstacle_handle : brinkman_cell_scratch_data.relevant_obstacle_handles)
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> local_obstacle_velocity =
        ObstacleType::get_velocity(brinkman_cell_scratch_data.relevant_obstacles,
                                   obstacle_handle,
                                   q_point);
      auto mask =
        brinkman_cell_scratch_data.mask_function(q_point,
                                                 brinkman_cell_scratch_data.relevant_obstacles,
                                                 obstacle_handle);
      momentum_penalty -= mask / brinkman_cell_scratch_data.data.permeability *
                          (fluid_momentum - w_q[0] * local_obstacle_velocity);
      energy_penalty = mask / brinkman_cell_scratch_data.data.permeability *
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
    const FluidStructureInteractionData<number>    &brinkman_penalization_data,
    typename BrinkmanPenalizationCellScratchData<dim, number, ObstacleType>::MaskFunctionType
      mask_function)
  : brinkman_cell_scratch_data(obstacle_handler,
                               brinkman_penalization_data,
                               std::move(mask_function))
{}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>::cell_operation(
  const dealii::MatrixFree<dim, number> &matrix_free,
  const unsigned int                     cell_batch_id)
{
  find_relevant_obstacles_in_cell_batch<dim, number, ObstacleType>(brinkman_cell_scratch_data,
                                                                   matrix_free,
                                                                   cell_batch_id);
}

template <int dim, typename number, typename ObstacleType>
auto
MeltPoolDG::BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>::quad_operation(
  const number,
  const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
  const ConservedVariablesType                              &w_q,
  const ConservedVariablesType                              &delta_w_q) -> ConservedVariablesType
{
  if (brinkman_cell_scratch_data.relevant_obstacle_handles.empty())
    return ConservedVariablesType();
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_momentum;
  for (int d = 0; d < dim; ++d)
    fluid_momentum[d] = w_q[d + 1];

  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_momentum_differential_change;
  for (int d = 0; d < dim; ++d)
    fluid_momentum_differential_change[d] = delta_w_q[d + 1];

  dealii::VectorizedArray<number>                         energy_penalty_differential_change(0.);
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum_penalty_differential_change;
  for (auto obstacle_handle : brinkman_cell_scratch_data.relevant_obstacle_handles)
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> local_obstacle_velocity =
        ObstacleType::get_velocity(brinkman_cell_scratch_data.relevant_obstacles,
                                   obstacle_handle,
                                   q_point);

      auto mask =
        brinkman_cell_scratch_data.mask_function(q_point,
                                                 brinkman_cell_scratch_data.relevant_obstacles,
                                                 obstacle_handle);

      energy_penalty_differential_change +=
        mask / brinkman_cell_scratch_data.data.permeability *
        (dealii::scalar_product(fluid_momentum, fluid_momentum) / (w_q[0] * w_q[0]) * delta_w_q[0] -
         2. / w_q[0] * dealii::scalar_product(fluid_momentum, fluid_momentum_differential_change) +
         dealii::scalar_product(local_obstacle_velocity, fluid_momentum_differential_change));

      for (int i = 0; i < dim; ++i)
        momentum_penalty_differential_change[i] -=
          mask / brinkman_cell_scratch_data.data.permeability *
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
