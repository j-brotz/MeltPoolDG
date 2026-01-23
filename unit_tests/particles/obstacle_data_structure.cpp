#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/affine_constraints.h>

#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <gtest_utils.hpp>

namespace
{
  constexpr int dim = 2;

  class ObstacleDataStructureFixture : public ::testing::Test
  {
  protected:
    using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;

    struct ParticleSpecification
    {
      dealii::Point<dim> location;
      double             radius;
      int                id;
    };

    ObstacleDataStructureFixture()
      : triangulation(
          MPI_COMM_WORLD,
          dealii::Triangulation<dim>::none,
          dealii::parallel::distributed::Triangulation<dim>::no_automatic_repartitioning)
      , mapping(1)
    {
      create_parallel_triangulation_with_specific_partitioning();

      obstacle_data_structure =
        std::make_unique<MeltPoolDG::ObstacleCompleteDomainSearch<dim, double, ObstacleType>>(
          triangulation, mapping);
    }

    void
    insert_particle(const ParticleSpecification &spec)
    {
      std::vector<double> properties(ObstacleType::n_obstacle_properties, 0);
      properties[ObstacleType::Properties::radius]      = spec.radius;
      properties[ObstacleType::Properties::particle_id] = spec.id;
      obstacle_data_structure->insert_obstacles(triangulation,
                                                std::vector<dealii::Point<dim, double>>{
                                                  spec.location},
                                                std::vector<std::vector<double>>{properties});
    }

    std::vector<int>
    extract_particle_ids(
      const std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> &handles) const
    {
      std::vector<int> ids;
      ids.reserve(handles.size());
      for (const auto &h : handles)
        ids.push_back(static_cast<int>(ObstacleType::get_property(
          obstacle_data_structure->get_particle_handler().get_property_pool(),
          h,
          ObstacleType::Properties::particle_id)));

      return ids;
    }

    dealii::parallel::distributed::Triangulation<dim> triangulation;
    dealii::MappingQ<dim>                             mapping;
    std::unique_ptr<MeltPoolDG::ObstacleCompleteDomainSearch<dim, double, ObstacleType>>
      obstacle_data_structure;

  private:
    void
    create_parallel_triangulation_with_specific_partitioning()
    {
      // Create the coarse mesh available on all processes
      dealii::GridGenerator::subdivided_hyper_cube(triangulation, 2, 0.0, 1.0);
      triangulation.refine_global(2);
    }
  };


  TEST_F(ObstacleDataStructureFixture, GetParticlesInCellSingleDistributedParticle)
  {
    MeltPoolDG::TestUtils::check_for_correct_mpi_process_count(MPI_COMM_WORLD, 4);

    insert_particle({.location = {0.35, 0.65}, .radius = 0.2, .id = 0});

    obstacle_data_structure->reinit();

    // We only search for the particles on the rank that actually owns the cell
    if (dealii::Utilities::MPI::this_mpi_process(triangulation.get_mpi_communicator()) == 3)
      {
        dealii::CellId cell_id_to_test("3_2:00");

        std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> relevant_particles =
          obstacle_data_structure->get_obstacles_in_cell(
            obstacle_data_structure->get_particle_handler().get_property_pool(),
            *triangulation.create_cell_iterator(cell_id_to_test));

        std::vector<int> expected_particle_ids{0};

        std::vector<int> computed_particle_id = extract_particle_ids(relevant_particles);

        EXPECT_EQ(expected_particle_ids, computed_particle_id);
      }
  }


  TEST_F(ObstacleDataStructureFixture, GetParticlesInCellSingleDistributedParticleOnCellEdge)
  {
    MeltPoolDG::TestUtils::check_for_correct_mpi_process_count(MPI_COMM_WORLD, 4);

    insert_particle({.location = {0.25, 0.65}, .radius = 0.25, .id = 0});

    obstacle_data_structure->reinit();

    // We only search for the particles on the rank that actually owns the cell
    if (dealii::Utilities::MPI::this_mpi_process(triangulation.get_mpi_communicator()) == 3)
      {
        dealii::CellId cell_id_to_test("3_2:02");

        std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> relevant_particles =
          obstacle_data_structure->get_obstacles_in_cell(
            obstacle_data_structure->get_particle_handler().get_property_pool(),
            *triangulation.create_cell_iterator(cell_id_to_test));

        std::vector<int> expected_particle_ids{0};

        std::vector<int> computed_particle_id = extract_particle_ids(relevant_particles);

        EXPECT_EQ(expected_particle_ids, computed_particle_id);
      }
  }

  TEST_F(ObstacleDataStructureFixture, GetParticlesInCellTwoParticles)
  {
    MeltPoolDG::TestUtils::check_for_correct_mpi_process_count(MPI_COMM_WORLD, 4);

    insert_particle({.location = {0.1875, 0.1875}, .radius = 0.1875, .id = 0});
    insert_particle({.location = {0.5625, 0.1875}, .radius = 0.2, .id = 1});
    insert_particle({.location = {0.5625, 0.6875}, .radius = 0.1, .id = 2});

    obstacle_data_structure->reinit();

    // We only search for the particles on the rank that actually owns the cell
    if (dealii::Utilities::MPI::this_mpi_process(triangulation.get_mpi_communicator()) == 0)
      {
        dealii::CellId cell_id_to_test("0_2:12");

        std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> relevant_particles =
          obstacle_data_structure->get_obstacles_in_cell(
            obstacle_data_structure->get_particle_handler().get_property_pool(),
            *triangulation.create_cell_iterator(cell_id_to_test));

        std::vector<int> expected_particle_ids{0, 1};

        std::vector<int> computed_particle_id = extract_particle_ids(relevant_particles);

        std::ranges::sort(computed_particle_id);

        EXPECT_EQ(expected_particle_ids, computed_particle_id);
      }
  }

  TEST_F(ObstacleDataStructureFixture, GetParticlesInCellBatch)
  {
    MeltPoolDG::TestUtils::check_for_correct_mpi_process_count(MPI_COMM_WORLD, 4);

    // Create a dummy matrix-free object
    dealii::DoFHandler<dim>           dof_handler(triangulation);
    dealii::AffineConstraints<double> constraints;
    dof_handler.distribute_dofs(dealii::FE_Q<dim>(1));
    dealii::QGauss<dim>             quadrature(1);
    dealii::MatrixFree<dim, double> matrix_free;
    matrix_free.reinit(mapping, dof_handler, constraints, quadrature);

    const auto check_for_expected_cells_in_batch =
      [&](const unsigned int          cell_batch_id,
          std::vector<dealii::CellId> expected_cell_ids,
          const unsigned int          mpi_rank) {
        if (dealii::Utilities::MPI::this_mpi_process(triangulation.get_mpi_communicator()) == 0)
          {
            std::vector<dealii::CellId> computed_cell_ids =
              std::invoke([&]() -> std::vector<dealii::CellId> {
                std::vector<dealii::CellId> cell_ids;
                cell_ids.reserve(matrix_free.n_active_entries_per_cell_batch(cell_batch_id));
                for (unsigned int j = 0;
                     j < matrix_free.n_active_entries_per_cell_batch(cell_batch_id);
                     ++j)
                  cell_ids.emplace_back(matrix_free.get_cell_iterator(cell_batch_id, j)->id());
                return cell_ids;
              });

            std::sort(expected_cell_ids.begin(), expected_cell_ids.end());
            std::sort(computed_cell_ids.begin(), computed_cell_ids.end());
            ASSERT_EQ(expected_cell_ids, computed_cell_ids);
          }
      };

#if defined(__AVX512F__) && DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 512
    constexpr unsigned cell_batch_of_interest = 0;
    std::vector<int>   expected_particle_ids{0, 1};
    check_for_expected_cells_in_batch(cell_batch_of_interest,
                                      {dealii::CellId("0_2:00"),
                                       dealii::CellId("0_2:01"),
                                       dealii::CellId("0_2:02"),
                                       dealii::CellId("0_2:03"),
                                       dealii::CellId("0_2:10"),
                                       dealii::CellId("0_2:11"),
                                       dealii::CellId("0_2:12"),
                                       dealii::CellId("0_2:13")},
                                      0);
#elif defined(__AVX__) && DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 256
    constexpr unsigned cell_batch_of_interest = 2;
    std::vector<int>   expected_particle_ids{0, 1};
    check_for_expected_cells_in_batch(cell_batch_of_interest,
                                      {dealii::CellId("0_2:10"),
                                       dealii::CellId("0_2:11"),
                                       dealii::CellId("0_2:12"),
                                       dealii::CellId("0_2:13")},
                                      0);
#elif (defined(__SSE2__) || defined(__ARM_NEON)) && DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 128
    constexpr unsigned cell_batch_of_interest = 3;
    std::vector<int>   expected_particle_ids{0, 1};
    check_for_expected_cells_in_batch(cell_batch_of_interest,
                                      {dealii::CellId("0_2:12"), dealii::CellId("0_2:13")},
                                      0);
#else
    constexpr unsigned cell_batch_of_interest = 1;
    std::vector<int>   expected_particle_ids{0};
    check_for_expected_cells_in_batch(cell_batch_of_interest, {dealii::CellId("0_2:01")}, 0);
#endif

    insert_particle({.location = {0.1875, 0.1875}, .radius = 0.1875, .id = 0});
    insert_particle({.location = {0.5625, 0.1875}, .radius = 0.125, .id = 1});
    insert_particle({.location = {0.5625, 0.6875}, .radius = 0.1, .id = 2});

    obstacle_data_structure->reinit();

    // We only search for the particles on the rank that actually owns the cell batch
    if (dealii::Utilities::MPI::this_mpi_process(triangulation.get_mpi_communicator()) == 0)
      {
        std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> relevant_particles =
          obstacle_data_structure->get_obstacles_in_cell_batch(
            obstacle_data_structure->get_particle_handler().get_property_pool(),
            matrix_free,
            cell_batch_of_interest);

        std::vector<int> expected_particle_ids{0, 1};

        std::vector<int> computed_particle_id = extract_particle_ids(relevant_particles);

        std::ranges::sort(computed_particle_id);

        EXPECT_EQ(expected_particle_ids, computed_particle_id);
      }
  }
} // namespace
