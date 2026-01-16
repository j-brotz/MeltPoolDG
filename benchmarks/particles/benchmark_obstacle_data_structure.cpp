
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include <memory>
#include <random>
#include <vector>


namespace
{

  /**
   * Generates @p n_total_particles reproducible random particle positions in the
   * unit square.
   */
  template <int dim, typename ObstacleType>
  std::pair<std::vector<dealii::Point<dim, double>>, std::vector<std::vector<double>>>
  generate_random_particles(const unsigned n_total_particles,
                            const double   min_particle_radius,
                            const double   max_particle_radius)
  {
    std::vector<dealii::Point<dim, double>> particle_positions;
    particle_positions.reserve(n_total_particles);
    std::vector<std::vector<double>> properties;
    properties.reserve(n_total_particles);

    std::mt19937                           gen(777);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> radius_dist(min_particle_radius, max_particle_radius);
    for (unsigned int i = 0; i < n_total_particles; ++i)
      {
        if constexpr (dim == 2)
          particle_positions.emplace_back(dist(gen), dist(gen));
        else if constexpr (dim == 3)
          particle_positions.emplace_back(dist(gen), dist(gen), dist(gen));
        else
          static_assert(dim == 2 || dim == 3, "Only dimensions 2 and 3 are supported.");

        std::vector<double> particle_properties(ObstacleType::n_obstacle_properties, 0);
        particle_properties[ObstacleType::Properties::radius]      = radius_dist(gen);
        particle_properties[ObstacleType::Properties::particle_id] = i;
        properties.push_back(particle_properties);
      }
    return {particle_positions, properties};
  }


  // TODO: How to deal with that?
  constexpr int      dim                 = 2;
  constexpr int      n_tria_refinements  = 6;
  constexpr double   min_particle_radius = 0.05;
  constexpr double   max_particle_radius = 0.1;
  constexpr unsigned n_particles         = 100;


  class ParticleFieldFixture : public benchmark::Fixture
  {
    // Prevent compiler warnings about hiding overloaded virtual functions (-Woverloaded-virtual).
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

  public:
    using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;

    struct Data
    {
    public:
      explicit Data(const benchmark::State &)
        : tria(MPI_COMM_WORLD)
        , mapping(1)
      {
        dealii::GridGenerator::hyper_cube(tria);
        tria.refine_global(n_tria_refinements);

        auto [particle_locations, particle_properties] =
          generate_random_particles<dim, ObstacleType>(n_particles,
                                                       min_particle_radius,
                                                       max_particle_radius);

        obstacle_data_structure =
          std::make_unique<MeltPoolDG::ObstacleCompleteDomainSearch<dim, double, ObstacleType>>(
            tria, mapping);

        obstacle_data_structure->insert_obstacles(tria, particle_locations, particle_properties);

        obstacle_data_structure->reinit();
      }

      dealii::parallel::distributed::Triangulation<dim> tria;
      dealii::MappingQ<dim>                             mapping;

      MeltPoolDG::ObstacleData<double> obstacle_data;
      std::unique_ptr<MeltPoolDG::ObstacleCompleteDomainSearch<dim, double, ObstacleType>>
        obstacle_data_structure;
    };

    std::unique_ptr<Data> data;

    void
    SetUp(benchmark::State &state) override
    {
      data = std::make_unique<Data>(state);
    }

    void
    TearDown(benchmark::State &) override
    {
      data.reset();
    }
  };


  BENCHMARK_F(ParticleFieldFixture, ObstaclesInCell)
  (benchmark::State &st)
  {
    std::vector<dealii::Triangulation<dim>::active_cell_iterator> locally_owned_active_cells;
    locally_owned_active_cells.reserve(data->tria.n_locally_owned_active_cells());
    for (const auto &cell : data->tria.active_cell_iterators())
      if (cell->is_locally_owned())
        locally_owned_active_cells.push_back(cell);

    for (auto _ : st)
      {
        for (const auto &cell : locally_owned_active_cells)
          {
            benchmark::DoNotOptimize(data->obstacle_data_structure->get_obstacles_in_cell(
              data->obstacle_data_structure->get_particle_handler().get_property_pool(), *cell));
          }
      }
  }


  BENCHMARK_F(ParticleFieldFixture, ObstaclesInCellBatch)
  (benchmark::State &st)
  {
    // Create a dummy matrix-free object
    dealii::DoFHandler<dim>           dof_handler(data->tria);
    dealii::AffineConstraints<double> constraints;
    dof_handler.distribute_dofs(dealii::FE_Q<dim>(1));
    dealii::QGauss<dim>             quadrature(1);
    dealii::MatrixFree<dim, double> matrix_free;
    matrix_free.reinit(data->mapping, dof_handler, constraints, quadrature);

    for (auto _ : st)
      {
        for (unsigned int cell_batch_id = 0; cell_batch_id < matrix_free.n_cell_batches();
             ++cell_batch_id)
          {
            benchmark::DoNotOptimize(data->obstacle_data_structure->get_obstacles_in_cell_batch(
              data->obstacle_data_structure->get_particle_handler().get_property_pool(),
              matrix_free,
              cell_batch_id));
          }
      }
  }
} // namespace

MPDG_BENCHMARK_MPI_MAIN;
