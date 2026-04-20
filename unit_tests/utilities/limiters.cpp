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
  using TensorType           = dealii::Tensor<1, n_components, double>;
} // namespace

/**
 * Given a collection of vectors where every single component is strictly positive, this test checks
 * that the TVD minmod operator selects the component-wise minimum magnitude among them.
 */
TEST(TVDMinmodTest, AllPositiveValues)
{
  std::vector<TensorType> values = {TensorType{{2.0, 3.0, 4.0}},
                                    TensorType{{1.0, 5.0, 6.0}},
                                    TensorType{{3.0, 2.0, 7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvd_minmod<n_components, TensorType, std::vector<TensorType>>(values);

  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 4.0);
}

/**
 * This test verifies that tvd_minmod returns zero when input signs are mixed.
 */
TEST(MinmodTest, MixedSignValues)
{
  std::vector<TensorType> values = {TensorType{{2.0, -3.0, 4.0}},
                                    TensorType{{-1.0, 5.0, -6.0}},
                                    TensorType{{3.0, -2.0, 7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvd_minmod<n_components, TensorType, std::vector<TensorType>>(values);

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
  std::vector<TensorType> values = {TensorType{{-2.0, -3.0, -4.0}},
                                    TensorType{{-1.0, -5.0, -6.0}},
                                    TensorType{{-3.0, -2.0, -7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvd_minmod<n_components, TensorType, std::vector<TensorType>>(values);

  EXPECT_DOUBLE_EQ(result[0], -1.0);
  EXPECT_DOUBLE_EQ(result[1], -2.0);
  EXPECT_DOUBLE_EQ(result[2], -4.0);
}

/**
 * This test verifies that tvd_minmod returns zero if any input component is zero.
 */
TEST(MinmodTest, ContainingZeroValues)
{
  std::vector<TensorType> values = {TensorType{{0.0, -3.0, 4.0}},
                                    TensorType{{-1.0, 0.0, -6.0}},
                                    TensorType{{3.0, -2.0, 0.0}}};

  auto result =
    MeltPoolDG::Utilities::tvd_minmod<n_components, TensorType, std::vector<TensorType>>(values);

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
  std::vector<double>     tvb_constant(3, 1.0);
  constexpr double        cell_size = 0.5;
  std::vector<TensorType> values    = {TensorType{{2.0, 3.0, 4.0}},
                                       TensorType{{1.0, 5.0, 6.0}},
                                       TensorType{{3.0, 2.0, 7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvb_minmod<double, n_components, TensorType, std::vector<TensorType>>(
      values, tvb_constant, cell_size);

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
  std::vector<double>     tvb_constant(3, 0.1);
  constexpr double        cell_size = 5.0;
  std::vector<TensorType> values    = {TensorType{{2.0, 3.0, 4.0}},
                                       TensorType{{1.0, 5.0, 6.0}},
                                       TensorType{{3.0, 2.0, 7.0}}};

  auto result =
    MeltPoolDG::Utilities::tvb_minmod<double, n_components, TensorType, std::vector<TensorType>>(
      values, tvb_constant, cell_size);

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

  // Set cell average values
  std::vector<TensorType> cell_average_values(triangulation.n_active_cells());
  TensorType              average_value{{1.0, 2.0, 3.0}};
  TensorType              average_value_modifier{{-1.0, -1.0, 1.0}};
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell_average_values[cell->active_cell_index()] = average_value;
      average_value += average_value_modifier;
    }

  dealii::Tensor<1, n_components, dealii::Tensor<1, 1, double>> average_cell_gradient;
  average_cell_gradient[0][0] = 1.0;
  average_cell_gradient[1][0] = -1.0;
  average_cell_gradient[2][0] = 2.0;

  {
    SCOPED_TRACE("Cell in the middle of the domain.");
    auto limited_value = MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(
      cell_average_values, ++triangulation.begin_active(), average_cell_gradient, limiter_data);

    EXPECT_DOUBLE_EQ(limited_value[0][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[0][1], -1.0);
    EXPECT_DOUBLE_EQ(limited_value[0][2], 2.0);
  }

  {
    SCOPED_TRACE("Cell at domain boundary.");
    average_cell_gradient[1][0] = -8.0;
    auto limited_value          = MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(
      cell_average_values, triangulation.begin_active(), average_cell_gradient, limiter_data);

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
  std::vector<TensorType> cell_average_values(triangulation.n_active_cells());
  TensorType              average_value{{1.0, 2.0, 3.0}};
  TensorType              average_value_modifier{{-1.0, -1.0, 1.0}};
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell_average_values[cell->active_cell_index()] = average_value;
      average_value += average_value_modifier;
    }

  dealii::Tensor<1, n_components, dealii::Tensor<1, dim, double>> average_cell_gradient;
  average_cell_gradient[0][0] = 1.0;
  average_cell_gradient[0][1] = 2.0;
  average_cell_gradient[1][0] = -1.0;
  average_cell_gradient[1][1] = -2.0;
  average_cell_gradient[2][0] = 2.0;
  average_cell_gradient[2][1] = 4.0;

  {
    SCOPED_TRACE("Cell in the middle of the domain.");
    auto limited_value =
      MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(cell_average_values,
                                                                triangulation.create_cell_iterator(
                                                                  dealii::CellId("0_2:03")),
                                                                average_cell_gradient,
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
                                                                average_cell_gradient,
                                                                limiter_data);

    EXPECT_DOUBLE_EQ(limited_value[0][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[0][1], -1.0);
    EXPECT_DOUBLE_EQ(limited_value[0][2], 2.0);

    EXPECT_DOUBLE_EQ(limited_value[1][0], 0.0);
    EXPECT_DOUBLE_EQ(limited_value[1][1], -2.0);
    EXPECT_DOUBLE_EQ(limited_value[1][2], 4.0);
  }
}
