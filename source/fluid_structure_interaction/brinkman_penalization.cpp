#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/particles/particle_accessor.h>

#include "meltpooldg/flow/flow_utils.hpp"
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <utility>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::BrinkmanObstacleForce<dim, number, ObstacleType>::BrinkmanObstacleForce(
  const ObstacleField<dim, number, ObstacleType> &obstacle_field,
  const VectorType                               &solution,
  const MatrixFreeContext<dim, number>           &matrix_free,
  const BrinkmanPenalizationData<number>         &data,
  FlowSolverType                                  flow_solver_type,
  const number                                    constant_density)
  : brinkman_penalization_data(data)
  , cell_obstacle_cache(obstacle_field)
  , matrix_free(matrix_free)
  , solution(solution)
  , flow_solver_type(flow_solver_type)
  , constant_density(constant_density)
{}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::BrinkmanObstacleForce<dim, number, ObstacleType>::local_apply_cell_compressible_flow(
  const dealii::MatrixFree<dim, number> &,
  dealii::Tensor<1, dim, number> &,
  const VectorType                     &solution,
  const std::pair<unsigned, unsigned>  &cell_range,
  dealii::Particles::PropertyPool<dim> &global_particle_properties) const
{
  FECellIntegrator<dim, dim + 2, number> phi(matrix_free.mf,
                                             matrix_free.dof_idx,
                                             matrix_free.quad_idx);

  for (unsigned cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.gather_evaluate(solution, dealii::EvaluationFlags::values);

      for (const unsigned int q : phi.quadrature_point_indices())
        {
          const auto [density, momentum, energy] =
            Flow::get_conserved_variables<dim>(phi.get_value(q));

          for (unsigned int src_handle = 0;
               src_handle < global_particle_properties.n_registered_slots();
               ++src_handle)
            {
              dealii::Tensor<1, dim, VectorizedArrayType>         force;
              dealii::Tensor<1, torque_size, VectorizedArrayType> torque;

              const auto mask = mask_function<dim, VectorizedArrayType, ObstacleType>(
                brinkman_penalization_data.mask_function_type,
                phi.quadrature_point(q),
                global_particle_properties,
                src_handle);
              force = mask / brinkman_penalization_data.permeability *
                      (momentum - density * ObstacleType::get_velocity(global_particle_properties,
                                                                       src_handle,
                                                                       phi.quadrature_point(q))) *
                      phi.JxW(q);

              if constexpr (dim == 2)
                {
                  auto vector_to_center_of_gravity =
                    ObstacleType::vector_to_center_of_gravity(phi.quadrature_point(q),
                                                              global_particle_properties,
                                                              src_handle);

                  torque[0] = -vector_to_center_of_gravity[0] * force[1] +
                              vector_to_center_of_gravity[1] * force[0];
                }
              if constexpr (dim == 3)
                {
                  torque = -dealii::cross_product_3d(
                    ObstacleType::vector_to_center_of_gravity(phi.quadrature_point(q),
                                                              global_particle_properties,
                                                              src_handle),
                    force);
                }

              dealii::Tensor<1, dim, number>         summed_force;
              dealii::Tensor<1, torque_size, number> summed_torque;

              for (int i = 0; i < dim; ++i)
                summed_force[i] = force[i].sum();
              for (unsigned i = 0; i < torque_size; ++i)
                summed_torque[i] = torque[i].sum();

              ObstacleType::accumulate_force(summed_force, global_particle_properties, src_handle);
              ObstacleType::accumulate_torque(summed_torque,
                                              global_particle_properties,
                                              src_handle);
            }
        }
    }
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::BrinkmanObstacleForce<dim, number, ObstacleType>::local_apply_cell_incompressible_flow(
  const dealii::MatrixFree<dim, number> &,
  dealii::Tensor<1, dim, number> &,
  const VectorType                     &velocity_solution,
  const std::pair<unsigned, unsigned>  &cell_range,
  dealii::Particles::PropertyPool<dim> &global_particle_properties) const
{
  FECellIntegrator<dim, dim, number> phi(matrix_free.mf, matrix_free.dof_idx, matrix_free.quad_idx);

  for (unsigned cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.gather_evaluate(velocity_solution, dealii::EvaluationFlags::values);

      for (const unsigned int q : phi.quadrature_point_indices())
        {
          dealii::Tensor<1, dim, VectorizedArrayType> fluid_velocity;

          // TODO: This is ugly
          if constexpr (dim == 1)
            fluid_velocity[0] = phi.get_value(q);
          else
            fluid_velocity = phi.get_value(q);
          for (unsigned int src_handle = 0;
               src_handle < global_particle_properties.n_registered_slots();
               ++src_handle)
            {
              dealii::Tensor<1, dim, VectorizedArrayType>         force;
              dealii::Tensor<1, torque_size, VectorizedArrayType> torque;

              const auto q_point = phi.quadrature_point(q);

              const auto mask = mask_function<dim, VectorizedArrayType, ObstacleType>(
                brinkman_penalization_data.mask_function_type,
                q_point,
                global_particle_properties,
                src_handle);

              force =
                mask / brinkman_penalization_data.permeability * constant_density *
                (fluid_velocity -
                 ObstacleType::get_velocity(global_particle_properties, src_handle, q_point)) *
                phi.JxW(q);

              if constexpr (dim == 2)
                {
                  auto vector_to_center_of_gravity =
                    ObstacleType::vector_to_center_of_gravity(q_point,
                                                              global_particle_properties,
                                                              src_handle);

                  torque[0] = -vector_to_center_of_gravity[0] * force[1] +
                              vector_to_center_of_gravity[1] * force[0];
                }
              if constexpr (dim == 3)
                {
                  torque =
                    -dealii::cross_product_3d(ObstacleType::vector_to_center_of_gravity(
                                                q_point, global_particle_properties, src_handle),
                                              force);
                }

              dealii::Tensor<1, dim, number>         summed_force;
              dealii::Tensor<1, torque_size, number> summed_torque;

              for (int i = 0; i < dim; ++i)
                summed_force[i] = force[i].sum();
              for (unsigned i = 0; i < torque_size; ++i)
                summed_torque[i] = torque[i].sum();

              ObstacleType::accumulate_force(summed_force, global_particle_properties, src_handle);
              ObstacleType::accumulate_torque(summed_torque,
                                              global_particle_properties,
                                              src_handle);
            }
        }
    }
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::BrinkmanObstacleForce<dim, number, ObstacleType>::add_load_to_obstacles(
  ObstacleField<dim, number, ObstacleType> &obstacle_field) const
{
  // We need a copy of the global particle properties in order to reset obstacle forces.
  auto global_particle_properties =
    obstacle_field.get_obstacle_data_structure().get_global_particle_properties();

  for (unsigned int src_handle = 0; src_handle < global_particle_properties.n_registered_slots();
       ++src_handle)
    ObstacleType::set_force(dealii::Tensor<1, dim, number>(),
                            global_particle_properties,
                            src_handle);

  // Step 1: Cell loop locally on each process to compute the local force and torque contribution to
  // each global particle for each MPI rank. The local force and torque results are accumulated in
  // the global_particle_properties variable.
  std::function<void(const dealii::MatrixFree<dim, number> &,
                     dealii::Tensor<1, dim, number> &,
                     const VectorType &,
                     const std::pair<unsigned, unsigned> &)>
    local_apply_cell = [&](const dealii::MatrixFree<dim, number> &,
                           dealii::Tensor<1, dim, number>      &force,
                           const VectorType                    &solution,
                           const std::pair<unsigned, unsigned> &cell_range) {
      if (flow_solver_type == FlowSolverType::compressible)
        local_apply_cell_compressible_flow(
          matrix_free.mf, force, solution, cell_range, global_particle_properties);
      else if (flow_solver_type == FlowSolverType::incompressible)
        {
          local_apply_cell_incompressible_flow(
            matrix_free.mf, force, solution, cell_range, global_particle_properties);
        }
      else
        AssertThrow(false, dealii::ExcMessage("Flow solver type not supported."));
    };

  dealii::Tensor<1, dim, number> particle_force_dummy;
  matrix_free.mf.cell_loop(local_apply_cell, particle_force_dummy, solution);

  // Step 3: Broadcast local force and torque results and accumulate them on all processes. This
  // results in the final fluid force and torque acting on the particles.
  for (unsigned int src_handle = 0; src_handle < global_particle_properties.n_registered_slots();
       ++src_handle)
    {
      dealii::Tensor<1, dim, number> local_force =
        ObstacleType::get_force(global_particle_properties, src_handle);
      dealii::Tensor<1, torque_size, number> local_torque =
        ObstacleType::get_torque(global_particle_properties, src_handle);

      dealii::Tensor<1, dim, number>         new_force;
      dealii::Tensor<1, torque_size, number> new_torque;

      for (int i = 0; i < dim; ++i)
        MPI_Allreduce(&local_force[i], &new_force[i], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      for (unsigned i = 0; i < torque_size; ++i)
        MPI_Allreduce(&local_torque[i], &new_torque[i], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      ObstacleType::set_force(new_force, global_particle_properties, src_handle);
      ObstacleType::set_torque(new_torque, global_particle_properties, src_handle);
    }

  // Step 4: Return result to actual local particles in obstacle field.
  for (auto &particle : obstacle_field.get_particle_handler())
    {
      for (unsigned int src_handle = 0;
           src_handle < global_particle_properties.n_registered_slots();
           ++src_handle)
        {
          if (particle.get_id() == global_particle_properties.get_properties(
                                     src_handle)[ObstacleType::Properties::particle_id])
            {
              ObstacleType::accumulate_force(ObstacleType::get_force(global_particle_properties,
                                                                     src_handle),
                                             particle);
              ObstacleType::accumulate_torque(ObstacleType::get_torque(global_particle_properties,
                                                                       src_handle),
                                              particle);
            }
        }
    }

  // We need to deregister all particles before the destructor of global_particle_properties is
  // called.
  for (unsigned int i = 0; i < global_particle_properties.n_registered_slots(); ++i)
    global_particle_properties.deregister_particle(i);
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
void
MeltPoolDG::BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>::cell_operation(
  const MatrixFreeContext<dim, number> &matrix_free,
  const unsigned int                    cell_batch_id)
{
  find_relevant_obstacles_in_cell_batch<dim, number, ObstacleType>(cell_obstacle_cache,
                                                                   matrix_free.mf,
                                                                   cell_batch_id);
}

template <int dim, typename number, typename ObstacleType>
auto
MeltPoolDG::BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>::quad_operation(
  const number,
  const dealii::Point<dim, VectorizedArrayType> &q_point,
  const ConservedVariablesType                  &w_q) -> ConservedVariablesType
{
  if (cell_obstacle_cache.relevant_obstacle_handles.empty())
    return ConservedVariablesType();

  const auto [density, momentum, energy] = Flow::get_conserved_variables<dim>(w_q);

  dealii::Tensor<1, dim, VectorizedArrayType> momentum_penalty;
  VectorizedArrayType                         energy_penalty = 0.;
  for (auto obstacle_handle : cell_obstacle_cache.relevant_obstacle_handles)
    {
      const dealii::Tensor<1, dim, VectorizedArrayType> local_obstacle_velocity =
        ObstacleType::get_velocity(cell_obstacle_cache.relevant_obstacles,
                                   obstacle_handle,
                                   q_point);
      auto mask = mask_function<dim, VectorizedArrayType, ObstacleType>(
        brinkman_penalization_data.mask_function_type,
        q_point,
        cell_obstacle_cache.relevant_obstacles,
        obstacle_handle);
      momentum_penalty -= mask / brinkman_penalization_data.permeability *
                          (momentum - density * local_obstacle_velocity);
      energy_penalty = mask / brinkman_penalization_data.permeability *
                       (momentum - density * local_obstacle_velocity) * momentum / density;
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
void
MeltPoolDG::BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>::cell_operation(
  const MatrixFreeContext<dim, number> &matrix_free,
  const unsigned int                    cell_batch_id)
{
  find_relevant_obstacles_in_cell_batch<dim, number, ObstacleType>(cell_obstacle_cache,
                                                                   matrix_free.mf,
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
  if (cell_obstacle_cache.relevant_obstacle_handles.empty())
    return ConservedVariablesType();

  const auto [density, momentum, energy] = Flow::get_conserved_variables<dim>(w_q);

  const auto [delta_density, delta_momentum, delta_energy] =
    Flow::get_conserved_variables<dim>(delta_w_q);

  VectorizedArrayType                         energy_penalty_differential_change(0.);
  dealii::Tensor<1, dim, VectorizedArrayType> momentum_penalty_differential_change;
  for (auto obstacle_handle : cell_obstacle_cache.relevant_obstacle_handles)
    {
      const dealii::Tensor<1, dim, VectorizedArrayType> local_obstacle_velocity =
        ObstacleType::get_velocity(cell_obstacle_cache.relevant_obstacles,
                                   obstacle_handle,
                                   q_point);

      auto mask = mask_function<dim, VectorizedArrayType, ObstacleType>(
        brinkman_penalization_data.mask_function_type,
        q_point,
        cell_obstacle_cache.relevant_obstacles,
        obstacle_handle);

      energy_penalty_differential_change +=
        mask / brinkman_penalization_data.permeability *
        (dealii::scalar_product(momentum, momentum) / (density * density) * delta_density -
         2. / w_q[0] * dealii::scalar_product(momentum, delta_momentum) +
         dealii::scalar_product(local_obstacle_velocity, delta_momentum));

      for (int i = 0; i < dim; ++i)
        momentum_penalty_differential_change[i] -=
          mask / brinkman_penalization_data.permeability *
          (delta_momentum[i] - local_obstacle_velocity[i] * delta_density[0]);
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
