#include <gtest/gtest.h>

#include <deal.II/base/tensor.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/utilities/limiters.hpp>
#include <meltpooldg/utilities/limiters.templates.hpp>

#include <vector>

namespace
{
  constexpr int n_components = 3;
  using TensorValueType      = dealii::Tensor<1, n_components, double>;
  template <int dim>
  using TensorGradientType = dealii::Tensor<1, n_components, dealii::Tensor<1, dim, double>>;
} // namespace

/**
 * Given a collection of vectors where every single component is strictly positive, this test checks
 * that the TVD minmod operator selects the component-wise minimum magnitude among them.
 */
TEST(TVDMinmodTest, AllPositiveValues)
{
  std::vector<TensorValueType> values = {TensorValueType{{2.0, 3.0, 4.0}},
                                         TensorValueType{{1.0, 5.0, 6.0}},
                                         TensorValueType{{3.0, 2.0, 7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvd_minmod<n_components, TensorValueType, std::vector<TensorValueType>>(
      values);

  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 4.0);
}

/**
 * This test verifies that tvd_minmod returns zero when input signs are mixed.
 */
TEST(MinmodTest, MixedSignValues)
{
  std::vector<TensorValueType> values = {TensorValueType{{2.0, -3.0, 4.0}},
                                         TensorValueType{{-1.0, 5.0, -6.0}},
                                         TensorValueType{{3.0, -2.0, 7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvd_minmod<n_components, TensorValueType, std::vector<TensorValueType>>(
      values);

  EXPECT_DOUBLE_EQ(result[0], 0.0);
  EXPECT_DOUBLE_EQ(result[1], 0.0);
  EXPECT_DOUBLE_EQ(result[2], 0.0);
}

/**
 * Given a collection of vectors where every single component is strictly negative, this test checks
 * that the TVD minmod operator selects the component-wise value that has the minimum absolute
 * value.
 */
TEST(MinmodTest, AllNegativeValues)
{
  std::vector<TensorValueType> values = {TensorValueType{{-2.0, -3.0, -4.0}},
                                         TensorValueType{{-1.0, -5.0, -6.0}},
                                         TensorValueType{{-3.0, -2.0, -7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvd_minmod<n_components, TensorValueType, std::vector<TensorValueType>>(
      values);

  EXPECT_DOUBLE_EQ(result[0], -1.0);
  EXPECT_DOUBLE_EQ(result[1], -2.0);
  EXPECT_DOUBLE_EQ(result[2], -4.0);
}

/**
 * This test verifies that tvd_minmod returns zero if any input component is zero.
 */
TEST(MinmodTest, ContainingZeroValues)
{
  std::vector<TensorValueType> values = {TensorValueType{{0.0, -3.0, 4.0}},
                                         TensorValueType{{-1.0, 0.0, -6.0}},
                                         TensorValueType{{3.0, -2.0, 0.0}}};

  auto result =
    MeltPoolDG::Utilities::tvd_minmod<n_components, TensorValueType, std::vector<TensorValueType>>(
      values);

  EXPECT_DOUBLE_EQ(result[0], 0.0);
  EXPECT_DOUBLE_EQ(result[1], 0.0);
  EXPECT_DOUBLE_EQ(result[2], 0.0);
}

/**
 * Check that the TVB modified minmod function returns the original value if all input values are
 * below the given slope limit multiplied with the cell size squared.
 */
TEST(TVBMinmodTest, AllBelowSlopeLimit)
{
  std::vector<double>                       tvb_constant(3, 100.0);
  constexpr double                          cell_size = 0.5;
  std::vector<dealii::Tensor<1, 3, double>> values    = {
    dealii::Tensor<1, n_components, double>{{2.0, 3.0, 8.0}},
    dealii::Tensor<1, n_components, double>{{1.0, 5.0, 6.0}},
    dealii::Tensor<1, n_components, double>{{3.0, 2.0, 7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvb_minmod<double,
                                      n_components,
                                      dealii::Tensor<1, n_components, double>,
                                      std::vector<dealii::Tensor<1, n_components, double>>>(
      values, tvb_constant, cell_size);

  EXPECT_DOUBLE_EQ(result[0], 2.0);
  EXPECT_DOUBLE_EQ(result[1], 3.0);
  EXPECT_DOUBLE_EQ(result[2], 8.0);
}

/**
 * Check that the TVB modified minmod function returns the TVD minmod value if all input values are
 * above the given slope limit multiplied with the cell size squared.
 */
TEST(TVBMinmodTest, AllAboveSlopeLimit)
{
  std::vector<double>          tvb_constant(3, 1.0);
  constexpr double             cell_size = 0.5;
  std::vector<TensorValueType> values    = {TensorValueType{{2.0, 3.0, 4.0}},
                                            TensorValueType{{1.0, 5.0, 6.0}},
                                            TensorValueType{{3.0, 2.0, 7.0}}};

  auto result = MeltPoolDG::Utilities::
    tvb_minmod<double, n_components, TensorValueType, std::vector<TensorValueType>>(values,
                                                                                    tvb_constant,
                                                                                    cell_size);

  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 4.0);
}

/**
 * Combines the above two tests by checking that the TVB modified minmod function returns the
 * original value for components that are below the slope limit and the TVD minmod value for
 * components that are above the slope limit if the input values contain a mix of values above and
 * below the slope limit.
 */
TEST(TVBMinmodTest, MixedBelowAndAboveSlopeLimit)
{
  std::vector<double>          tvb_constant(3, 0.1);
  constexpr double             cell_size = 5.0;
  std::vector<TensorValueType> values    = {TensorValueType{{2.0, 3.0, 4.0}},
                                            TensorValueType{{1.0, 5.0, 6.0}},
                                            TensorValueType{{3.0, 2.0, 7.0}}};

  auto result = MeltPoolDG::Utilities::
    tvb_minmod<double, n_components, TensorValueType, std::vector<TensorValueType>>(values,
                                                                                    tvb_constant,
                                                                                    cell_size);

  EXPECT_DOUBLE_EQ(result[0], 2.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 4.0);
}

/**
 * Evaluates the function `compute_minmod_type_limited_slopes()` across a 1D refined hyper-cube,
 * verifying correct neighbor evaluation for both an internal cell (middle of domain) and a boundary
 * cell.
 */
TEST(MinmodTypeLimiterSlopes, 1D)
{
  constexpr int dim = 1;

  MeltPoolDG::Utilities::LimiterData<double> limiter_data;
  limiter_data.type = MeltPoolDG::Utilities::LimiterType::tvd_minmod;

  // Create the triangulation
  dealii::Triangulation<dim> triangulation;
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);

  // Set cell average values and gradients
  std::vector<std::pair<TensorValueType, TensorGradientType<dim>>> cell_average_values(
    triangulation.n_active_cells());
  TensorValueType         average_value{{1.0, 2.0, 3.0}};
  TensorGradientType<dim> average_cell_gradient;
  average_cell_gradient[0][0] = 1.0;
  average_cell_gradient[1][0] = -1.0;
  average_cell_gradient[2][0] = 2.0;
  TensorValueType average_value_modifier{{-1.0, -1.0, 1.0}};
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell_average_values[cell->active_cell_index()] =
        std::make_pair(average_value, average_cell_gradient);
      average_value += average_value_modifier;
    }

  {
    SCOPED_TRACE("Cell in the middle of the domain.");
    auto limited_value =
      MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(cell_average_values,
                                                                ++triangulation.begin_active(),
                                                                limiter_data);

    EXPECT_DOUBLE_EQ(limited_value[0][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[0][1], -1.0);
    EXPECT_DOUBLE_EQ(limited_value[0][2], 2.0);
  }

  {
    SCOPED_TRACE("Cell at domain boundary.");
    average_cell_gradient[1][0] = -8.0;
    cell_average_values[triangulation.begin_active()->active_cell_index()].second =
      average_cell_gradient;
    auto limited_value =
      MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(cell_average_values,
                                                                triangulation.begin_active(),
                                                                limiter_data);

    EXPECT_DOUBLE_EQ(limited_value[0][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[0][1], -4.0);
    EXPECT_DOUBLE_EQ(limited_value[0][2], 2.0);
  }
}

/**
 * Evaluates the function `compute_minmod_type_limited_slopes()` across a 2D refined hyper-cube,
 * verifying correct neighbor evaluation for both an internal cell (middle of domain) and a boundary
 * cell (in the corner of the domain).
 */
TEST(MinmodTypeLimiterSlopes, 2D)
{
  constexpr int dim = 2;

  MeltPoolDG::Utilities::LimiterData<double> limiter_data;
  limiter_data.type = MeltPoolDG::Utilities::LimiterType::tvd_minmod;

  // Create the triangulation
  dealii::Triangulation<dim> triangulation;
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);

  // Set cell average values
  std::vector<std::pair<TensorValueType, TensorGradientType<dim>>> cell_average_values(
    triangulation.n_active_cells());
  TensorValueType         average_value{{1.0, 2.0, 3.0}};
  TensorValueType         average_value_modifier{{-1.0, -1.0, 1.0}};
  TensorGradientType<dim> average_cell_gradient;
  average_cell_gradient[0][0] = 1.0;
  average_cell_gradient[0][1] = 2.0;
  average_cell_gradient[1][0] = -1.0;
  average_cell_gradient[1][1] = -2.0;
  average_cell_gradient[2][0] = 2.0;
  average_cell_gradient[2][1] = 4.0;
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell_average_values[cell->active_cell_index()] =
        std::make_pair(average_value, average_cell_gradient);
      average_value += average_value_modifier;
    }

  {
    SCOPED_TRACE("Cell in the middle of the domain.");
    auto limited_value =
      MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(cell_average_values,
                                                                triangulation.create_cell_iterator(
                                                                  dealii::CellId("0_2:03")),
                                                                limiter_data);

    EXPECT_DOUBLE_EQ(limited_value[0][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[0][1], -1.0);
    EXPECT_DOUBLE_EQ(limited_value[0][2], 2.0);

    EXPECT_DOUBLE_EQ(limited_value[1][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[1][1], -2.0);
    EXPECT_DOUBLE_EQ(limited_value[1][2], 4.0);
  }

  {
    SCOPED_TRACE("Cell at domain boundary.");
    auto limited_value =
      MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(cell_average_values,
                                                                triangulation.create_cell_iterator(
                                                                  dealii::CellId("0_2:00")),
                                                                limiter_data);

    EXPECT_DOUBLE_EQ(limited_value[0][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[0][1], -1.0);
    EXPECT_DOUBLE_EQ(limited_value[0][2], 2.0);

    EXPECT_DOUBLE_EQ(limited_value[1][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[1][1], -2.0);
    EXPECT_DOUBLE_EQ(limited_value[1][2], 4.0);
  }
}

/**
 * Unit test verifying the 2D Minmod-type slope limiter on a locally refined mesh.
 *
 * The domain initially consist of five corse cells in the x-direction and one cell in the
 * y-direction. The rightmost coarse cell is refined once isotopically as illustrated below.
 *
 * +-------+-------+-------+-------+---+---+
 * |       |       |       |       | 2 | 3 |
 * |  C0   |  C1   |  C2   |  C3   +---+---+
 * |       |       |       |       | 0 | 1 |
 * +-------+-------+-------+-------+---+---+
 *
 * The following two scenarios are evaluated in the test:
 * 1. Finer cell neighbor: Check that the limiter correctly works if a neighbor is finer than the
 * current cell. The check is performed using the cell C3 in the above sketch.
 * 2. Coarser cell neighbor: Check that the limiter correctly works if a neighbor is coarser than
 * the current cell. The check is performed using the cell 2 in the above sketch.
 */
TEST(MinmodTypeLimiterSlopes, 2D_local_mesh_refinement)
{
  constexpr int dim = 2;

  MeltPoolDG::Utilities::LimiterData<double> limiter_data;
  limiter_data.type = MeltPoolDG::Utilities::LimiterType::tvd_minmod;

  // Create the triangulation
  dealii::Triangulation<dim> triangulation;
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                    std::vector<unsigned int>{5, 1},
                                                    dealii::Point<dim>(0.0, 0.0),
                                                    dealii::Point<dim>(5.0, 1.0));
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->center()[0] > 4)
      cell->set_refine_flag();

  triangulation.prepare_coarsening_and_refinement();
  triangulation.execute_coarsening_and_refinement();

  // Set cell average values
  std::vector<std::pair<TensorValueType, TensorGradientType<dim>>> cell_average_values(
    triangulation.n_active_cells());
  TensorValueType         average_value{{1.0, 2.0, 3.0}};
  TensorValueType         average_value_modifier{{-1.0, -1.0, 1.0}};
  TensorGradientType<dim> average_cell_gradient;
  average_cell_gradient[0][0] = 1.0;
  average_cell_gradient[0][1] = 2.0;
  average_cell_gradient[1][0] = -1.0;
  average_cell_gradient[1][1] = -2.0;
  average_cell_gradient[2][0] = 2.0;
  average_cell_gradient[2][1] = 4.0;
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell_average_values[cell->active_cell_index()] =
        std::make_pair(average_value, average_cell_gradient);
      average_value += average_value_modifier;
    }

  {
    SCOPED_TRACE("Finer cell neighbors");
    auto limited_value =
      MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(cell_average_values,
                                                                triangulation.create_cell_iterator(
                                                                  dealii::CellId("3_0:0")),
                                                                limiter_data);

    EXPECT_DOUBLE_EQ(limited_value[0][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[0][1], -1.0);
    EXPECT_DOUBLE_EQ(limited_value[0][2], 1.0);

    EXPECT_DOUBLE_EQ(limited_value[1][0], 2.0);
    EXPECT_DOUBLE_EQ(limited_value[1][1], -2.0);
    EXPECT_DOUBLE_EQ(limited_value[1][2], 4.0);
  }

  {
    SCOPED_TRACE("Coarser cell neighbors");

    // Explicitly modify some values to ensure that the minmod limiter will return the values
    // obtained from the averaging with the coarser neighbor.
    cell_average_values[triangulation.create_cell_iterator(dealii::CellId("4_1:2"))
                          ->active_cell_index()]
      .second[1][0] = -10.0;
    cell_average_values[triangulation.create_cell_iterator(dealii::CellId("4_1:2"))
                          ->active_cell_index()]
      .second[2][0] = 10.0;

    cell_average_values[triangulation.create_cell_iterator(dealii::CellId("4_1:2"))
                          ->neighbor(1)
                          ->active_cell_index()]
      .first[1] = -20.0;
    cell_average_values[triangulation.create_cell_iterator(dealii::CellId("4_1:2"))
                          ->neighbor(1)
                          ->active_cell_index()]
      .first[2] = 20.0;

    auto limited_value =
      MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(cell_average_values,
                                                                triangulation.create_cell_iterator(
                                                                  dealii::CellId("4_1:2")),
                                                                limiter_data);

    EXPECT_DOUBLE_EQ(limited_value[0][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[0][1], -5.5);
    EXPECT_DOUBLE_EQ(limited_value[0][2], 5.0);

    EXPECT_DOUBLE_EQ(limited_value[1][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[1][1], -2.0);
    EXPECT_DOUBLE_EQ(limited_value[1][2], 4.0);
  }
}
