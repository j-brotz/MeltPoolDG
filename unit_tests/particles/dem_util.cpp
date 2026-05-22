#include <gtest/gtest.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/particles/dem_util.hpp>

#include <algorithm>
#include <vector>

TEST(GetNeighborCells, TwoDimensionalInnerCell)
{
  constexpr int dim = 2;

  dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(2);

  dealii::CellId cell_to_check("0_2:12");

  MeltPoolDG::LevelCellCache<dim> level_cell_cache;
  level_cell_cache.reinit(triangulation, 2);

  // We only perform the check on the process which actually owns the cell
  if (triangulation.contains_cell(cell_to_check))
    {
      typename dealii::Triangulation<dim>::cell_iterator cell_to_check_it =
        triangulation.create_cell_iterator(cell_to_check);
      if (cell_to_check_it->is_locally_owned())
        {
          // Make the expected neighbors
          std::vector<dealii::CellId> expected_neighbors;
          expected_neighbors.reserve(9);
          expected_neighbors.emplace_back("0_2:12");
          expected_neighbors.emplace_back("0_2:10");
          expected_neighbors.emplace_back("0_2:11");
          expected_neighbors.emplace_back("0_2:13");
          expected_neighbors.emplace_back("0_2:01");
          expected_neighbors.emplace_back("0_2:03");
          expected_neighbors.emplace_back("0_2:21");
          expected_neighbors.emplace_back("0_2:30");
          expected_neighbors.emplace_back("0_2:31");

          // Compute the neighbors using the function
          std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> neighbors =
            level_cell_cache.get_neighboring_cells(cell_to_check_it);

          // Cast the neighbors to CellIds
          std::vector<dealii::CellId> computed_neighbors;
          computed_neighbors.reserve(neighbors.size());
          for (const auto &neighbor : neighbors)
            computed_neighbors.push_back(neighbor->id());

          // Sort the computed and reference vector for comparison
          std::ranges::sort(computed_neighbors, [](const auto &a, const auto &b) { return a < b; });
          std::ranges::sort(expected_neighbors, [](const auto &a, const auto &b) { return a < b; });

          // Compare the computed and reference vectors
          EXPECT_EQ(computed_neighbors, expected_neighbors);
        }
    }
}

TEST(GetNeighborCells, ThreeDimensionalInnerCell)
{
  constexpr int dim = 3;

  dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(2);

  dealii::CellId cell_to_check("0_2:16");

  MeltPoolDG::LevelCellCache<dim> level_cell_cache;
  level_cell_cache.reinit(triangulation, 2);

  // We only perform the check on the process which actually owns the cell
  if (triangulation.contains_cell(cell_to_check))
    {
      typename dealii::Triangulation<dim>::cell_iterator cell_to_check_it =
        triangulation.create_cell_iterator(cell_to_check);
      if (cell_to_check_it->is_locally_owned())
        {
          // Make the expected neighbors
          std::vector<dealii::CellId> expected_neighbors;
          expected_neighbors.reserve(27);
          expected_neighbors.emplace_back("0_2:10");
          expected_neighbors.emplace_back("0_2:11");
          expected_neighbors.emplace_back("0_2:12");
          expected_neighbors.emplace_back("0_2:13");
          expected_neighbors.emplace_back("0_2:14");
          expected_neighbors.emplace_back("0_2:15");
          expected_neighbors.emplace_back("0_2:16");
          expected_neighbors.emplace_back("0_2:17");

          expected_neighbors.emplace_back("0_2:01");
          expected_neighbors.emplace_back("0_2:03");
          expected_neighbors.emplace_back("0_2:05");
          expected_neighbors.emplace_back("0_2:07");

          expected_neighbors.emplace_back("0_2:21");
          expected_neighbors.emplace_back("0_2:25");

          expected_neighbors.emplace_back("0_2:30");
          expected_neighbors.emplace_back("0_2:31");
          expected_neighbors.emplace_back("0_2:34");
          expected_neighbors.emplace_back("0_2:35");

          expected_neighbors.emplace_back("0_2:41");
          expected_neighbors.emplace_back("0_2:43");

          expected_neighbors.emplace_back("0_2:50");
          expected_neighbors.emplace_back("0_2:51");
          expected_neighbors.emplace_back("0_2:52");
          expected_neighbors.emplace_back("0_2:53");

          expected_neighbors.emplace_back("0_2:61");

          expected_neighbors.emplace_back("0_2:70");
          expected_neighbors.emplace_back("0_2:71");


          // Compute the neighbors using the function
          std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> neighbors =
            level_cell_cache.get_neighboring_cells(cell_to_check_it);

          // Cast the neighbors to CellIds
          std::vector<dealii::CellId> computed_neighbors;
          computed_neighbors.reserve(neighbors.size());
          for (const auto &neighbor : neighbors)
            computed_neighbors.push_back(neighbor->id());

          // Sort the computed and reference vector for comparison
          std::ranges::sort(computed_neighbors, [](const auto &a, const auto &b) { return a < b; });
          std::ranges::sort(expected_neighbors, [](const auto &a, const auto &b) { return a < b; });

          // Compare the computed and reference vectors
          EXPECT_EQ(computed_neighbors, expected_neighbors);
        }
    }
}

TEST(GetNeighborCells, ThreeDimensionalBoundaryCell)
{
  constexpr int dim = 3;

  dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(2);

  dealii::CellId cell_to_check("0_2:10");

  MeltPoolDG::LevelCellCache<dim> level_cell_cache;
  level_cell_cache.reinit(triangulation, 2);

  // We only perform the check on the processes which know about the cell
  if (triangulation.contains_cell(cell_to_check))
    {
      typename dealii::Triangulation<dim>::cell_iterator cell_to_check_it =
        triangulation.create_cell_iterator(cell_to_check);
      if (cell_to_check_it->is_locally_owned())
        {
          // Make the expected neighbors
          std::vector<dealii::CellId> expected_neighbors;
          expected_neighbors.reserve(12);
          expected_neighbors.emplace_back("0_2:10");
          expected_neighbors.emplace_back("0_2:11");
          expected_neighbors.emplace_back("0_2:12");
          expected_neighbors.emplace_back("0_2:13");
          expected_neighbors.emplace_back("0_2:14");
          expected_neighbors.emplace_back("0_2:15");
          expected_neighbors.emplace_back("0_2:16");
          expected_neighbors.emplace_back("0_2:17");

          expected_neighbors.emplace_back("0_2:01");
          expected_neighbors.emplace_back("0_2:03");
          expected_neighbors.emplace_back("0_2:05");
          expected_neighbors.emplace_back("0_2:07");

          // Compute the neighbors using the function
          std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> neighbors =
            level_cell_cache.get_neighboring_cells(cell_to_check_it);

          // Cast the neighbors to CellIds
          std::vector<dealii::CellId> computed_neighbors;
          computed_neighbors.reserve(neighbors.size());
          for (const auto &neighbor : neighbors)
            computed_neighbors.push_back(neighbor->id());

          // Sort the computed and reference vector for comparison
          std::ranges::sort(computed_neighbors, [](const auto &a, const auto &b) { return a < b; });
          std::ranges::sort(expected_neighbors, [](const auto &a, const auto &b) { return a < b; });

          // Compare the computed and reference vectors
          EXPECT_EQ(computed_neighbors, expected_neighbors);
        }
    }
}

TEST(GetNeighborCells, TwoDimensionalBoundaryCell)
{
  constexpr int dim = 2;

  dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(2);

  dealii::CellId cell_to_check("0_2:10");

  MeltPoolDG::LevelCellCache<dim> level_cell_cache;
  level_cell_cache.reinit(triangulation, 2);


  // We only perform the check on the process which actually owns the cell
  if (triangulation.contains_cell(cell_to_check))
    {
      typename dealii::Triangulation<dim>::cell_iterator cell_to_check_it =
        triangulation.create_cell_iterator(cell_to_check);
      if (cell_to_check_it->is_locally_owned())
        {
          // Make the expected neighbors
          std::vector<dealii::CellId> expected_neighbors;
          expected_neighbors.reserve(6);
          expected_neighbors.emplace_back("0_2:01");
          expected_neighbors.emplace_back("0_2:10");
          expected_neighbors.emplace_back("0_2:11");
          expected_neighbors.emplace_back("0_2:03");
          expected_neighbors.emplace_back("0_2:12");
          expected_neighbors.emplace_back("0_2:13");

          // Compute the neighbors using the function
          std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> neighbors =
            level_cell_cache.get_neighboring_cells(cell_to_check_it);

          // Cast the neighbors to CellIds
          std::vector<dealii::CellId> computed_neighbors;
          computed_neighbors.reserve(neighbors.size());
          for (const auto &neighbor : neighbors)
            computed_neighbors.push_back(neighbor->id());

          // Sort the computed and reference vector for comparison
          std::ranges::sort(computed_neighbors, [](const auto &a, const auto &b) { return a < b; });
          std::ranges::sort(expected_neighbors, [](const auto &a, const auto &b) { return a < b; });

          // Compare the computed and reference vectors
          EXPECT_EQ(computed_neighbors, expected_neighbors);
        }
    }
}

TEST(GetNeighborCells, TwoDimensionalInnerCellAtRefinementBoundary)
{
  constexpr int dim = 2;

  // Setup the grid
  dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(1);
  for (auto cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned() and cell->id() != dealii::CellId("0_1:3"))
        {
          cell->set_refine_flag();
        }
    }
  triangulation.prepare_coarsening_and_refinement();
  triangulation.execute_coarsening_and_refinement();

  MeltPoolDG::LevelCellCache<dim> level_cell_cache;
  level_cell_cache.reinit(triangulation, 2);

  dealii::CellId cell_to_check("0_2:21");

  // We only perform the check on the process which actually owns the cell
  if (triangulation.contains_cell(cell_to_check))
    {
      typename dealii::Triangulation<dim>::cell_iterator cell_to_check_it =
        triangulation.create_cell_iterator(cell_to_check);
      if (cell_to_check_it->is_locally_owned())
        {
          // Make the expected neighbors
          std::vector<dealii::CellId> expected_neighbors;
          expected_neighbors.reserve(8);
          expected_neighbors.emplace_back("0_2:02");
          expected_neighbors.emplace_back("0_2:03");
          expected_neighbors.emplace_back("0_2:12");
          expected_neighbors.emplace_back("0_2:20");
          expected_neighbors.emplace_back("0_2:21");
          expected_neighbors.emplace_back("0_2:22");
          expected_neighbors.emplace_back("0_2:23");
          expected_neighbors.emplace_back("0_1:3");

          // Compute the neighbors using the function
          std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> neighbors =
            level_cell_cache.get_neighboring_cells(cell_to_check_it);

          // Cast the neighbors to CellIds
          std::vector<dealii::CellId> computed_neighbors;
          computed_neighbors.reserve(neighbors.size());
          for (const auto &neighbor : neighbors)
            computed_neighbors.push_back(neighbor->id());

          // Sort the computed and reference vector for comparison
          std::ranges::sort(computed_neighbors, [](const auto &a, const auto &b) { return a < b; });
          std::ranges::sort(expected_neighbors, [](const auto &a, const auto &b) { return a < b; });
          std::cout << triangulation.n_cells(2) << std::endl;
          std::cout << triangulation.contains_cell(dealii::CellId("0_2:12")) << std::endl;

          EXPECT_EQ(computed_neighbors, expected_neighbors);
        }
    }
}