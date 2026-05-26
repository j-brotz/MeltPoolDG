#include <deal.II/base/array_view.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#include <meltpooldg/fluid_structure_interaction/stokes_law.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <numbers>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::StokesLawSphericalParticleForce<dim, number, ObstacleType>::
  StokesLawSphericalParticleForce(const VectorType                     &solution,
                                  const MatrixFreeContext<dim, number> &matrix_free,
                                  const number                          dynamic_viscosity)
  : matrix_free(matrix_free)
  , solution(solution)
  , dynamic_viscosity(dynamic_viscosity)
{}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::StokesLawSphericalParticleForce<dim, number, ObstacleType>::add_load_to_obstacles(
  ObstacleField<dim, number, ObstacleType> &obstacle_field) const
{
  for (DEMParticleAccessor<dim, number> &particle : obstacle_field.locally_owned_particle_range())
    {
      dealii::FEPointEvaluation<dim + 2, dim> fe_point_eval(
        *matrix_free.mf.get_mapping_info().mapping,
        matrix_free.mf.get_dof_handler(matrix_free.dof_idx).get_fe(),
        dealii::UpdateFlags::update_values);
      dealii::Point<dim, number> coordinates_on_unit_cell =
        matrix_free.mf.get_mapping_info().mapping->transform_real_to_unit_cell(
          particle.get_surrounding_cell(), particle.get_location());

      fe_point_eval.reinit(particle.get_surrounding_cell(),
                           dealii::ArrayView<dealii::Point<dim, number>>(coordinates_on_unit_cell));
      auto dof_cell = particle.get_surrounding_cell()->as_dof_handler_iterator(
        matrix_free.mf.get_dof_handler(matrix_free.dof_idx));
      dealii::Vector<number> dof_values(dof_cell->get_fe().n_dofs_per_cell());
      dof_cell->get_dof_values(solution, dof_values);
      fe_point_eval.evaluate(dof_values, dealii::EvaluationFlags::values);

      dealii::Tensor<1, dim + 2, number> w = fe_point_eval.get_value(0);
      dealii::Tensor<1, dim, number>     fluid_velocity;
      for (unsigned i = 0; i < dim; ++i)
        fluid_velocity[i] = w[i + 1] / w[0];

      dealii::Tensor<1, dim> particle_force = 6. * std::numbers::pi * particle.radius() *
                                              (fluid_velocity - particle.get_linear_velocity()) *
                                              dynamic_viscosity;
      particle.add_force(particle_force);
    }
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::StokesLawFluidForce<dim, number, ObstacleType>::StokesLawFluidForce(
  const dealii::LinearAlgebra::distributed::Vector<number> &solution,
  ObstacleField<dim, number, ObstacleType>                 &obstacle_handler,
  const MatrixFreeContext<dim, number>                     &matrix_free,
  const number                                              dynamic_viscosity)
  : solution(solution)
  , dynamic_viscosity(dynamic_viscosity)
  , matrix_free(matrix_free)
  , obstacle_handler(obstacle_handler)
{}

template <int dim, typename number, typename ObstacleType>
auto
MeltPoolDG::StokesLawFluidForce<dim, number, ObstacleType>::value(
  const number,
  const unsigned int,
  const boost::container::small_vector<dealii::TriaIterator<dealii::CellAccessor<dim>>,
                                       dealii::VectorizedArray<double>::size()> &cell_iterators,
  const dealii::Point<dim, dealii::VectorizedArray<number>> &,
  const ConservedVariablesType &) -> ConservedVariablesType
{
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> cell_penalty_force;
  for (unsigned int i = 0; i < cell_iterators.size(); ++i)
    {
      const auto &cell = cell_iterators[i];
      if (cell->state() == dealii::IteratorState::valid)
        {
          const number cell_volume = cell->measure();
          boost::container::small_vector<
            MeltPoolDG::DEMParticleAccessor<dim, number>,
            MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::
                max_particles_per_active_cell *
              8>
            particles;
          obstacle_handler.get_obstacles_in_cell(cell, particles);

          for (const DEMParticleAccessor<dim, number> &particle : particles)
            {
              auto [particle_surrounding_active_cell, coordinates_on_unit_cell] =
                dealii::GridTools::find_active_cell_around_point(
                  *matrix_free.mf.get_mapping_info().mapping,
                  matrix_free.mf.get_dof_handler(matrix_free.dof_idx).get_triangulation(),
                  particle.get_location());

              dealii::FEPointEvaluation<dim + 2, dim> fe_point_eval(
                *matrix_free.mf.get_mapping_info().mapping,
                matrix_free.mf.get_dof_handler(matrix_free.dof_idx).get_fe(),
                dealii::UpdateFlags::update_values);

              fe_point_eval.reinit(particle_surrounding_active_cell,
                                   dealii::ArrayView<dealii::Point<dim, number>>(
                                     coordinates_on_unit_cell));
              auto dof_cell = particle_surrounding_active_cell->as_dof_handler_iterator(
                matrix_free.mf.get_dof_handler(matrix_free.dof_idx));
              dealii::Vector<number> dof_values(dof_cell->get_fe().n_dofs_per_cell());
              dof_cell->get_dof_values(solution, dof_values);
              fe_point_eval.evaluate(dof_values, dealii::EvaluationFlags::values);

              dealii::Tensor<1, dim + 2, number> conserved_variables = fe_point_eval.get_value(0);

              // compute fluid velocity at particle location from corresponding conserved variables
              dealii::Tensor<1, dim, number> fluid_velocity;
              for (unsigned i = 0; i < dim; ++i)
                fluid_velocity[i] = conserved_variables[i + 1] / conserved_variables[0];

              // compute force contribution according to Stokes' law
              dealii::Tensor<1, dim, number> individual_particle_force =
                -6. * std::numbers::pi * particle.radius() *
                (fluid_velocity - particle.get_linear_velocity()) * dynamic_viscosity;

              for (int j = 0; j < dim; ++j)
                cell_penalty_force[j][i] += individual_particle_force[j] / cell_volume;
            }
        }
    }

  ConservedVariablesType fluid_force;
  for (int d = 0; d < dim; ++d)
    fluid_force[d + 1] = cell_penalty_force[d];
  return fluid_force;
}

template struct MeltPoolDG::
  StokesLawSphericalParticleForce<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  StokesLawSphericalParticleForce<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  StokesLawSphericalParticleForce<3, double, MeltPoolDG::SphericalParticle<3, double>>;


template struct MeltPoolDG::
  StokesLawFluidForce<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  StokesLawFluidForce<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  StokesLawFluidForce<3, double, MeltPoolDG::SphericalParticle<3, double>>;
