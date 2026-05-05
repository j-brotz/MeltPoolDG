#include <gtest/gtest.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/heat/laser_data.hpp>

#include <algorithm>
#include <array>
#include <cmath>

template <int dim>
void
expect_direction_near(const dealii::Tensor<1, dim>  &direction,
                      const std::array<double, dim> &expected)
{
  constexpr double rel_tol = 1e-7;
  constexpr double abs_tol = 1e-6;

  for (unsigned int d = 0; d < dim; ++d)
    EXPECT_NEAR(direction[d], expected[d], std::max(abs_tol, std::abs(rel_tol * expected[d])));
}

TEST(LaserDataTest, DefaultDirectionRotatedIn2D)
{
  using namespace MeltPoolDG::Heat;

  LaserData<double>        laser;
  dealii::ParameterHandler prm;

  laser.add_parameters(prm);

  prm.enter_subsection("laser");
  {
    // Default 2D direction before rotation is (0, -1).
    prm.set("beam rotation angle", 90.);
  }
  prm.leave_subsection();

  laser.post(2);

  const auto direction = laser.get_beam_direction<2>();

  // Rotating (0, -1) by 90° gives (1, 0).
  expect_direction_near<2>(direction, {{1.0, 0.0}});
}

TEST(LaserDataTest, CustomDirectionRotatedIn2D)
{
  using namespace MeltPoolDG::Heat;

  LaserData<double>        laser;
  dealii::ParameterHandler prm;

  laser.add_parameters(prm);

  prm.enter_subsection("laser");
  {
    prm.set("beam direction", "1, 0");
    prm.set("beam rotation angle", 90.);
  }
  prm.leave_subsection();

  laser.post(2);

  const auto direction = laser.get_beam_direction<2>();

  // Rotating (1, 0) by 90° gives (0, 1).
  expect_direction_near<2>(direction, {{0.0, 1.0}});
}

TEST(LaserDataTest, DefaultDirectionRotatedIn3DWithDefaultAxis)
{
  using namespace MeltPoolDG::Heat;

  LaserData<double>        laser;
  dealii::ParameterHandler prm;

  laser.add_parameters(prm);

  prm.enter_subsection("laser");
  {
    // Default 3D direction before rotation is (0, 0, -1).
    // Default 3D rotation axis is (0, -1, 0).
    prm.set("beam rotation angle", 90.);
  }
  prm.leave_subsection();

  laser.post(3);

  const auto direction = laser.get_beam_direction<3>();

  // Rotating (0, 0, -1) around (0, -1, 0) by 90° gives (1, 0, 0).
  expect_direction_near<3>(direction, {{1.0, 0.0, 0.0}});
}

TEST(LaserDataTest, CustomDirectionRotatedIn3DWithCustomAxis)
{
  using namespace MeltPoolDG::Heat;

  LaserData<double>        laser;
  dealii::ParameterHandler prm;

  laser.add_parameters(prm);

  prm.enter_subsection("laser");
  {
    prm.set("beam direction", "1, 0, 0");
    prm.set("beam rotation axis", "0, 0, 1");
    prm.set("beam rotation angle", 90.);
  }
  prm.leave_subsection();

  laser.post(3);

  const auto direction = laser.get_beam_direction<3>();

  // Rotating (1, 0, 0) around z by 90° gives (0, 1, 0).
  expect_direction_near<3>(direction, {{0.0, 1.0, 0.0}});
}
