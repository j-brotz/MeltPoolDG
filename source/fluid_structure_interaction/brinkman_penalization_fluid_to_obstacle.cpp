#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_fluid_to_obstacle.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_util.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <utility>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::BrinkmanObstacleForce<dim, number, ObstacleType>::BrinkmanObstacleForce(
  const VectorType                                                &solution,
  const dealii::MatrixFree<dim, number>                           &mf,
  const unsigned                                                   dof_idx,
  const unsigned                                                   quad_idx,
  BrinkmanPenalizationCellScratchData<dim, number, ObstacleType> &&scratch_data)
  : brinkman_scratch_data(scratch_data)
  , matrix_free(mf)
  , dof_idx(dof_idx)
  , quad_idx(quad_idx)
  , solution(solution)
{}
template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::BrinkmanObstacleForce<dim, number, ObstacleType>::add_load_to_obstacles(
  ObstacleField<dim, number, ObstacleType> &obstacle_field) const
{
  constexpr unsigned torque_size = ObstacleType::size_angular_velocity;

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
    local_apply_cell = [&](const dealii::MatrixFree<dim, number> &mf,
                           dealii::Tensor<1, dim, number> &,
                           const VectorType                    &solution,
                           const std::pair<unsigned, unsigned> &cell_range) {
      FECellIntegrator<dim, dim + 2, number> phi(mf, dof_idx, quad_idx);

      for (unsigned cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(solution, dealii::EvaluationFlags::values);

          for (const unsigned int q : phi.quadrature_point_indices())
            {
              const auto                                              w_q = phi.get_value(q);
              dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_momentum;
              for (int d = 0; d < dim; ++d)
                fluid_momentum[d] = w_q[d + 1];
              for (unsigned int src_handle = 0;
                   src_handle < global_particle_properties.n_registered_slots();
                   ++src_handle)
                {
                  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>         force;
                  dealii::Tensor<1, torque_size, dealii::VectorizedArray<number>> torque;

                  const auto mask = brinkman_scratch_data.mask_function(phi.quadrature_point(q),
                                                                        global_particle_properties,
                                                                        src_handle);
                  force           = mask / brinkman_scratch_data.data.permeability *
                          (fluid_momentum -
                           w_q[0] * ObstacleType::get_velocity(global_particle_properties,
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

                  ObstacleType::accumulate_force(summed_force,
                                                 global_particle_properties,
                                                 src_handle);
                  ObstacleType::accumulate_torque(summed_torque,
                                                  global_particle_properties,
                                                  src_handle);
                }
            }
        }
    };

  dealii::Tensor<1, dim, number> particle_force_dummy;
  matrix_free.cell_loop(local_apply_cell, particle_force_dummy, solution);

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

template struct MeltPoolDG::
  BrinkmanObstacleForce<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  BrinkmanObstacleForce<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  BrinkmanObstacleForce<3, double, MeltPoolDG::SphericalParticle<3, double>>;
