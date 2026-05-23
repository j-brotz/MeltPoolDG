#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include "meltpooldg/particles/contact_forces_data.hpp"
#include "meltpooldg/time_integration/time_stepping_data.hpp"
#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include <memory>
#include <vector>

constexpr int dim = 3;

void
setup_rectangular_triangulation(dealii::Triangulation<dim>      &tria,
                                const std::vector<unsigned int> &n_base_cells,
                                const dealii::Point<dim>        &bottom_left,
                                const dealii::Point<dim>        &top_right,
                                unsigned int                     n_global_refinements)
{
  dealii::GridGenerator::subdivided_hyper_rectangle(tria, n_base_cells, bottom_left, top_right);

  tria.refine_global(n_global_refinements);
}

MeltPoolDG::ObstacleField<dim, double, MeltPoolDG::SphericalParticle<dim, double>>
setup_particle_field(const dealii::Triangulation<dim>       &triangulation,
                     const MeltPoolDG::ObstacleData<double> &data,
                     dealii::TimerOutput                    &timer)
{
  MeltPoolDG::ObstacleField<dim, double, MeltPoolDG::SphericalParticle<dim, double>> obstacle_field(
    data, triangulation, dealii::MappingQ1<dim>(), timer);

  return obstacle_field;
}

MeltPoolDG::SphericalParticleContactForce<dim, double, MeltPoolDG::SphericalParticle<dim, double>>
setup_contact_forces_with_ground_wall(
  const MeltPoolDG::SphericalParticleContactData<double>  &data,
  const MeltPoolDG::TimeIntegration::TimeIterator<double> &time_iterator)
{
  MeltPoolDG::SphericalParticleContactForce<dim, double, MeltPoolDG::SphericalParticle<dim, double>>
    contact_force(data, time_iterator);

  dealii::Tensor<1, dim, double> ground_wall_normal;
  ground_wall_normal[dim - 1] = 1.;
  contact_force.attach_wall(
    std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(dealii::Point<dim>(),
                                                                    ground_wall_normal));
  return contact_force;
}

static void
BM_ParticleBedOnGround(benchmark::State &state)
{
  dealii::parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD,
    dealii::Triangulation<dim>::MeshSmoothing::none,
    dealii::parallel::distributed::Triangulation<dim>::Settings::construct_multigrid_hierarchy);
  setup_rectangular_triangulation(triangulation,
                                  std::vector<unsigned int>{2, 2, 4},
                                  dealii::Point<dim>(0., 0., 0.),
                                  dealii::Point<dim>(2e-4, 2e-4, 4e-4),
                                  4);

  dealii::TimerOutput              timer(std::cout,
                            dealii::TimerOutput::OutputFrequency::never,
                            dealii::TimerOutput::wall_times);
  MeltPoolDG::ObstacleData<double> obstacle_data;
  obstacle_data.obstacle_state_input_file = "./particle_locations/28_particles.csv";
  auto obstacle_field = setup_particle_field(triangulation, obstacle_data, timer);

  MeltPoolDG::TimeIntegration::TimeSteppingData<double> time_stepping_data;
  MeltPoolDG::TimeIntegration::TimeIterator<double>     time_iterator(time_stepping_data);

  auto contact_force =
    setup_contact_forces_with_ground_wall(MeltPoolDG::SphericalParticleContactData<double>(),
                                          time_iterator);

  for (auto _ : state)
    {
      contact_force.add_load_to_obstacles(obstacle_field);
    }
}

BENCHMARK(BM_ParticleBedOnGround);

MPDG_BENCHMARK_MPI_MAIN;