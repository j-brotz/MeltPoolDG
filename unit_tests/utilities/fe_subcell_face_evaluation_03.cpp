/**
 * @file fe_subcell_face_evaluation_03.cpp
 * @brief Parallel unit tests for the reconstructed values on the inner and boundary faces of the
 * DG cells computed by FESubcellFaceEvaluation::get_face_value(). The tests cover the first and the
 * second order reconstruction, faces at the partition boundary, periodic faces, and faces at the
 * domain boundary.
 */

#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_subcell_face_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/limiter_data.hpp>
#include <meltpooldg/utilities/subcell_matrix_free.templates.hpp>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "../test_utils/test_functions.hpp"
#include "../test_utils/test_generators.hpp"
#include "../test_utils/utils.hpp"

using namespace MeltPoolDG;

namespace
{
  using VectorType               = dealii::LinearAlgebra::distributed::Vector<double>;
  using VectorizedArrayType      = dealii::VectorizedArray<double>;
  constexpr unsigned int n_lanes = VectorizedArrayType::size();

  constexpr double tolerance = 1e-10;

  /**
   * Parameters of a single test configuration.
   */
  template <int dim_, unsigned int degree_, int n_components_>
  struct TestConfiguration
  {
    static constexpr int          dim          = dim_;
    static constexpr unsigned int degree       = degree_;
    static constexpr int          n_components = n_components_;
  };
} // namespace

/**
 * Test fixture setting up a DG discretization on a distributed, anisotropic box with the same
 * number of cells in each direction. The cell sizes differ per direction. The boundary is
 * colorized, i.e. the boundary id of the lower and upper boundary in direction d is 2d and 2d+1,
 * and all faces of the locally owned cells are held by the MatrixFree object.
 */
template <typename Config>
class FESubcellFaceEvaluationReconstructionTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = Config::dim;
  static constexpr unsigned int degree       = Config::degree;
  static constexpr int          n_components = Config::n_components;

  static constexpr int n_cells_1d = dim == 2 ? 6 : 4;

  FESubcellFaceEvaluationReconstructionTest()
    : triangulation(MPI_COMM_WORLD)
    , scratch_data(MPI_COMM_WORLD, 0, true)
    , fe_data(FiniteElementType::FE_DGQ, degree)
  {}

  /**
   * Create the mesh and the matrix-free data structures. If @p periodic_in_x is true, the lower and
   * the upper boundary in x-direction are periodic.
   */
  void
  setup(const bool periodic_in_x)
  {
    std::vector<unsigned int> repetitions(dim, n_cells_1d);
    dealii::Point<dim>        upper_right;
    for (unsigned int d = 0; d < dim; ++d)
      {
        upper_right[d] = 1. + 0.5 * d;
        cell_size[d]   = upper_right[d] / n_cells_1d;
      }
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, repetitions, dealii::Point<dim>(), upper_right, /*colorize=*/true);

    if (periodic_in_x)
      {
        std::vector<dealii::GridTools::PeriodicFacePair<
          typename dealii::parallel::distributed::Triangulation<dim>::cell_iterator>>
          periodic_faces;
        dealii::GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
        triangulation.add_periodicity(periodic_faces);
      }

    dof_handler.reinit(triangulation);
    FiniteElementUtils::distribute_dofs<dim, n_components>(fe_data, dof_handler);
    constraints.close();

    scratch_data.set_mapping(FiniteElementUtils::create_mapping<dim>(fe_data));
    scratch_data.attach_quadrature(FiniteElementUtils::create_quadrature<dim>(fe_data));
    scratch_data.attach_dof_handler(dof_handler);
    scratch_data.attach_constraint_matrix(constraints);
    scratch_data.create_partitioning();
    scratch_data.build(true, true, true, false, true);
  }

  dealii::parallel::distributed::Triangulation<dim> triangulation;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::AffineConstraints<double>                 constraints;
  ScratchData<dim, dim, double>                     scratch_data;
  FiniteElementData                                 fe_data;
  std::array<double, dim>                           cell_size;
};

using TestConfigs = ::testing::Types<TestConfiguration<2, 1, 1>,
                                     TestConfiguration<2, 3, 3>,
                                     TestConfiguration<3, 1, 1>,
                                     TestConfiguration<3, 2, 2>>;

TYPED_TEST_SUITE(FESubcellFaceEvaluationReconstructionTest, TestConfigs);

/**
 * This test checks that the first order reconstruction returns the subcell values as face values.
 * The test sets up arbitrary DoF values and evaluates the interior and the exterior side of all
 * inner faces as well as the boundary faces with the default reconstruction order. The face values
 * returned by get_face_value() are then checked to be bitwise identical to the subcell values
 * returned by get_value().
 */
TYPED_TEST(FESubcellFaceEvaluationReconstructionTest, FirstOrderFaceValuesAreSubcellValues)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  for (unsigned int i = 0; i < solution.locally_owned_size(); ++i)
    solution.local_element(i) = TestUtils::arbitrary_value<double>();
  solution.update_ghost_values();

  FESubcellFaceEvaluation<dim, n_components, double> phi_m(subcell_matrix_free, true);
  FESubcellFaceEvaluation<dim, n_components, double> phi_p(subcell_matrix_free, false);

  for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);
      phi_p.gather_evaluate(solution, dealii::EvaluationFlags::values);

      for (const unsigned int s : phi_m.subcell_face_indices())
        {
          TestUtils::expect_near(phi_m.get_face_value(s, phi_p),
                                 phi_m.get_value(s),
                                 matrix_free.n_active_entries_per_face_batch(face),
                                 /*abs_tol=*/0.);
          TestUtils::expect_near(phi_p.get_face_value(s, phi_m),
                                 phi_p.get_value(s),
                                 matrix_free.n_active_entries_per_face_batch(face),
                                 /*abs_tol=*/0.);
        }
    }

  for (unsigned int face = matrix_free.n_inner_face_batches();
       face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
       ++face)
    {
      phi_m.reinit(face);
      phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);

      for (const unsigned int s : phi_m.subcell_face_indices())
        TestUtils::expect_near(phi_m.get_face_value(s),
                               phi_m.get_value(s),
                               matrix_free.n_active_entries_per_face_batch(face),
                               /*abs_tol=*/0.);
    }
}

/**
 * This test checks that the second order reconstruction of the values on the inner faces is exact
 * for a linear function for all limiter functions. For a linear function, both one-sided slopes of
 * a subcell coincide, i.e. r = 1, and all limiter functions satisfy phi(1) = 1, such that the
 * reconstructed values equal the linear function at the subcell face locations. The test sets up
 * the subcell values to the linear function at the subcell centers on all cells. The reconstructed
 * values on the interior and the exterior side of each subcell face of all inner faces, including
 * those at the partition boundary, are then checked to equal the linear function at the subcell
 * face location.
 */
TYPED_TEST(FESubcellFaceEvaluationReconstructionTest, InnerFaceValuesOfLinearFunctionAreExact)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const TestUtils::LinearFunction<dim, n_components> linear_function;

  // Set the subcell values to the linear function at the subcell locations. Interpolating the
  // linear function would give its values at the cell quadrature points instead, which differ from
  // the subcell locations.
  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  FESubcellEvaluation<dim, n_components, double> subcells(subcell_matrix_free);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      for (const unsigned int subcell : subcells.subcell_indices())
        subcells.submit_value(subcell, linear_function.value(subcells.subcell_location(subcell)));
      subcells.integrate();
      subcells.set_dof_values(solution);
    }
  solution.update_ghost_values();

  for (const Utilities::LimiterFunctionType limiter_function_type :
       {Utilities::LimiterFunctionType::tvd_minmod,
        Utilities::LimiterFunctionType::superbee,
        Utilities::LimiterFunctionType::muscl,
        Utilities::LimiterFunctionType::harmonic})
    {
      SCOPED_TRACE(limiter_function_type._to_string());

      // The boundary value function is not needed on inner faces.
      FESubcellFaceEvaluation<dim, n_components, double> phi_m(
        subcell_matrix_free, true, {}, 2, limiter_function_type);
      FESubcellFaceEvaluation<dim, n_components, double> phi_p(
        subcell_matrix_free, false, {}, 2, limiter_function_type);

      for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
          phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);
          phi_p.gather_evaluate(solution, dealii::EvaluationFlags::values);

          for (const unsigned int s : phi_m.subcell_face_indices())
            {
              const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type
                expected_value = linear_function.value(phi_m.subcell_face_location(s));

              TestUtils::expect_near(phi_m.get_face_value(s, phi_p),
                                     expected_value,
                                     matrix_free.n_active_entries_per_face_batch(face),
                                     tolerance);
              TestUtils::expect_near(phi_p.get_face_value(s, phi_m),
                                     expected_value,
                                     matrix_free.n_active_entries_per_face_batch(face),
                                     tolerance);
            }
        }
    }
}

/**
 * This test checks that the second order reconstruction of the values on the boundary faces is
 * exact for a linear function for all limiter functions, see the previous test. Contrary to inner
 * faces, the value across the face is the boundary value located on the boundary face. The test
 * sets up the subcell values to the linear function at the subcell centers on all cells, and the
 * boundary value function returns the linear function at the given point on the boundary face. The
 * reconstructed values on all boundary faces are then checked to equal the linear function at the
 * subcell face location.
 */
TYPED_TEST(FESubcellFaceEvaluationReconstructionTest, BoundaryFaceValuesOfLinearFunctionAreExact)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const TestUtils::LinearFunction<dim, n_components> linear_function;

  const typename FESubcellFaceEvaluation<dim, n_components, double>::BoundaryValueFunctionType
    boundary_value_function =
      [&linear_function](
        const dealii::Point<dim, VectorizedArrayType> &location,
        const dealii::Tensor<1, dim, VectorizedArrayType> &,
        const dealii::types::boundary_id,
        const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type &) ->
    typename FESubcellFaceEvaluation<dim, n_components, double>::value_type {
      return linear_function.value(location);
    };

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  FESubcellEvaluation<dim, n_components, double> subcells(subcell_matrix_free);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      for (const unsigned int subcell : subcells.subcell_indices())
        subcells.submit_value(subcell, linear_function.value(subcells.subcell_location(subcell)));
      subcells.integrate();
      subcells.set_dof_values(solution);
    }
  solution.update_ghost_values();

  unsigned int n_boundary_face_batches = 0;
  for (const Utilities::LimiterFunctionType limiter_function_type :
       {Utilities::LimiterFunctionType::tvd_minmod,
        Utilities::LimiterFunctionType::superbee,
        Utilities::LimiterFunctionType::muscl,
        Utilities::LimiterFunctionType::harmonic})
    {
      SCOPED_TRACE(limiter_function_type._to_string());

      FESubcellFaceEvaluation<dim, n_components, double> phi_m(
        subcell_matrix_free, true, boundary_value_function, 2, limiter_function_type);

      for (unsigned int face = matrix_free.n_inner_face_batches();
           face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
           ++face)
        {
          ++n_boundary_face_batches;

          phi_m.reinit(face);
          phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);

          for (const unsigned int s : phi_m.subcell_face_indices())
            TestUtils::expect_near(phi_m.get_face_value(s),
                                   linear_function.value(phi_m.subcell_face_location(s)),
                                   matrix_free.n_active_entries_per_face_batch(face),
                                   tolerance);
        }
    }

  // Sanity check to ensure that the test actually checked some boundary faces.
  EXPECT_GT(dealii::Utilities::MPI::sum(n_boundary_face_batches, MPI_COMM_WORLD), 0u);
}

/**
 * This test checks that the second order reconstruction is also exact across periodic faces, where
 * the distance of the two subcells adjacent to a subcell face must not be computed from their
 * locations. The test sets up a linear solution that is evaluated at x - L instead of x in the
 * right half of the domain, with L the domain length in x-direction. When gluing the right end of
 * the domain to the left one, the solution is thus linear across the periodic faces and jumps in
 * the middle of the domain instead, which is not reached by the stencils of the periodic faces. The
 * reconstructed values on both sides of the periodic faces are then checked to equal the linear
 * function at the subcell face location shifted in the same way.
 */
TYPED_TEST(FESubcellFaceEvaluationReconstructionTest, FaceValuesAcrossPeriodicFacesAreExact)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(true);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const TestUtils::LinearFunction<dim, n_components> linear_function;
  const double domain_length = TestFixture::n_cells_1d * this->cell_size[0];

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  FESubcellEvaluation<dim, n_components, double> subcells(subcell_matrix_free);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      for (const unsigned int subcell : subcells.subcell_indices())
        {
          dealii::Point<dim, VectorizedArrayType> location = subcells.subcell_location(subcell);
          for (unsigned int lane = 0; lane < n_lanes; ++lane)
            if (location[0][lane] > 0.5 * domain_length)
              location[0][lane] -= domain_length;
          subcells.submit_value(subcell, linear_function.value(location));
        }
      subcells.integrate();
      subcells.set_dof_values(solution);
    }
  solution.update_ghost_values();

  // The boundary value function is not needed on inner faces, which include the periodic faces.
  FESubcellFaceEvaluation<dim, n_components, double> phi_m(
    subcell_matrix_free, true, {}, 2, Utilities::LimiterFunctionType::tvd_minmod);
  FESubcellFaceEvaluation<dim, n_components, double> phi_p(
    subcell_matrix_free, false, {}, 2, Utilities::LimiterFunctionType::tvd_minmod);

  unsigned int n_periodic_face_lanes = 0;
  for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);
      phi_p.gather_evaluate(solution, dealii::EvaluationFlags::values);

      for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face); ++lane)
        {
          const std::pair<typename dealii::DoFHandler<dim>::cell_iterator, unsigned int>
            cell_and_face_no = matrix_free.get_face_iterator(face, lane);
          if (not cell_and_face_no.first->has_periodic_neighbor(cell_and_face_no.second))
            continue;
          ++n_periodic_face_lanes;

          for (const unsigned int s : phi_m.subcell_face_indices())
            {
              // Shift the face locations in the same way as the solution, which maps both ends
              // of the domain in x-direction onto x = 0.
              dealii::Point<dim, VectorizedArrayType> location_m = phi_m.subcell_face_location(s);
              dealii::Point<dim, VectorizedArrayType> location_p = phi_p.subcell_face_location(s);
              if (location_m[0][lane] > 0.5 * domain_length)
                location_m[0][lane] -= domain_length;
              if (location_p[0][lane] > 0.5 * domain_length)
                location_p[0][lane] -= domain_length;

              const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type
                face_value_m = phi_m.get_face_value(s, phi_p);
              const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type
                face_value_p = phi_p.get_face_value(s, phi_m);
              const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type
                expected_value_m = linear_function.value(location_m);
              const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type
                expected_value_p = linear_function.value(location_p);

              if constexpr (n_components == 1)
                {
                  EXPECT_NEAR(face_value_m[lane], expected_value_m[lane], tolerance);
                  EXPECT_NEAR(face_value_p[lane], expected_value_p[lane], tolerance);
                }
              else
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    EXPECT_NEAR(face_value_m[c][lane], expected_value_m[c][lane], tolerance);
                    EXPECT_NEAR(face_value_p[c][lane], expected_value_p[c][lane], tolerance);
                  }
            }
        }
    }

  // Sanity check to ensure that the test actually checked some periodic faces.
  EXPECT_GT(dealii::Utilities::MPI::sum(n_periodic_face_lanes, MPI_COMM_WORLD), 0u);
}

/**
 * This test checks that the second order reconstruction with the minmod limiter function does not
 * create new extrema on the inner faces, i.e. the reconstructed value on each side of a subcell
 * face lies between the values of the two subcells adjacent to it. The distance to the subcell
 * across the face is at least half the width of the subcell, such that this bound holds for the
 * minmod limiter function also on the non-uniform subcells. The test sets up arbitrary DoF values,
 * such that the one-sided slopes of the subcells frequently differ in sign and magnitude. The
 * reconstructed values on both sides of each subcell face of all inner faces are then checked to
 * lie between the values of the two adjacent subcells.
 */
TYPED_TEST(FESubcellFaceEvaluationReconstructionTest, InnerFaceValuesWithMinmodAreBounded)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  for (unsigned int i = 0; i < solution.locally_owned_size(); ++i)
    solution.local_element(i) = TestUtils::arbitrary_value<double>();
  solution.update_ghost_values();

  // The boundary value function is not needed on inner faces.
  FESubcellFaceEvaluation<dim, n_components, double> phi_m(
    subcell_matrix_free, true, {}, 2, Utilities::LimiterFunctionType::tvd_minmod);
  FESubcellFaceEvaluation<dim, n_components, double> phi_p(
    subcell_matrix_free, false, {}, 2, Utilities::LimiterFunctionType::tvd_minmod);

  for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);
      phi_p.gather_evaluate(solution, dealii::EvaluationFlags::values);

      for (const unsigned int s : phi_m.subcell_face_indices())
        {
          // Store the values as tensors to access components and lanes in the same way for a
          // single and multiple components.
          dealii::Tensor<1, n_components, VectorizedArrayType> value_m;
          dealii::Tensor<1, n_components, VectorizedArrayType> value_p;
          dealii::Tensor<1, n_components, VectorizedArrayType> face_value_m;
          dealii::Tensor<1, n_components, VectorizedArrayType> face_value_p;
          if constexpr (n_components == 1)
            {
              value_m[0]      = phi_m.get_value(s);
              value_p[0]      = phi_p.get_value(s);
              face_value_m[0] = phi_m.get_face_value(s, phi_p);
              face_value_p[0] = phi_p.get_face_value(s, phi_m);
            }
          else
            {
              value_m      = phi_m.get_value(s);
              value_p      = phi_p.get_value(s);
              face_value_m = phi_m.get_face_value(s, phi_p);
              face_value_p = phi_p.get_face_value(s, phi_m);
            }

          for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face);
               ++lane)
            for (unsigned int c = 0; c < n_components; ++c)
              {
                const double lower_bound = std::min(value_m[c][lane], value_p[c][lane]) - tolerance;
                const double upper_bound = std::max(value_m[c][lane], value_p[c][lane]) + tolerance;

                EXPECT_GE(face_value_m[c][lane], lower_bound);
                EXPECT_LE(face_value_m[c][lane], upper_bound);
                EXPECT_GE(face_value_p[c][lane], lower_bound);
                EXPECT_LE(face_value_p[c][lane], upper_bound);
              }
        }
    }
}

/**
 * This test checks that constructing an object with an unsupported reconstruction order throws an
 * exception. Since the check uses AssertThrow, it is also performed in release mode.
 */
TYPED_TEST(FESubcellFaceEvaluationReconstructionTest, UnsupportedReconstructionOrderThrows)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  SubcellMatrixFree<dim, double> subcell_matrix_free;
  subcell_matrix_free.reinit(this->scratch_data.get_matrix_free(), 0, 0);

  EXPECT_THROW(
    (FESubcellFaceEvaluation<dim, n_components, double>(subcell_matrix_free, true, {}, 0)),
    dealii::ExceptionBase);
  EXPECT_THROW(
    (FESubcellFaceEvaluation<dim, n_components, double>(subcell_matrix_free, true, {}, 3)),
    dealii::ExceptionBase);
}

/**
 * This test checks that invalid usage of get_face_value() triggers an assertion in debug mode.
 * Assertions should be triggered by calling it before evaluating the values, with an object of the
 * same side of the face, with an object of another face batch, by calling the boundary face
 * variant on an inner face, and by calling the inner face variant on a boundary face. Since
 * assertions are disabled in release mode, the test is skipped there.
 */
TYPED_TEST(FESubcellFaceEvaluationReconstructionTest, InvalidUsageAsserts)
{
  if constexpr (!dealii::running_in_debug_mode())
    GTEST_SKIP() << "Assertions are only checked in debug mode.";

  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const typename FESubcellFaceEvaluation<dim, n_components, double>::BoundaryValueFunctionType
    boundary_value_function =
      [](const dealii::Point<dim, VectorizedArrayType> &,
         const dealii::Tensor<1, dim, VectorizedArrayType> &,
         const dealii::types::boundary_id,
         const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type &value) ->
    typename FESubcellFaceEvaluation<dim, n_components, double>::value_type { return value; };

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  dealii::VectorTools::interpolate(this->scratch_data.get_mapping(),
                                   this->dof_handler,
                                   TestUtils::LinearFunction<dim, n_components>(),
                                   solution);
  solution.update_ghost_values();

  ASSERT_GE(matrix_free.n_inner_face_batches(), 2u);

  TestUtils::ScopedThrowOnAssert scoped_throw_on_assert;

  FESubcellFaceEvaluation<dim, n_components, double> phi_m(subcell_matrix_free,
                                                           true,
                                                           boundary_value_function,
                                                           2);
  FESubcellFaceEvaluation<dim, n_components, double> phi_p(subcell_matrix_free,
                                                           false,
                                                           boundary_value_function,
                                                           2);

  phi_m.reinit(0);
  phi_p.reinit(0);
  EXPECT_THROW(phi_m.get_face_value(0, phi_p), dealii::ExceptionBase);

  phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);
  phi_p.gather_evaluate(solution, dealii::EvaluationFlags::values);
  EXPECT_THROW(phi_m.get_face_value(0, phi_m), dealii::ExceptionBase);
  EXPECT_THROW(phi_m.get_face_value(0), dealii::ExceptionBase);

  phi_p.reinit(1);
  phi_p.gather_evaluate(solution, dealii::EvaluationFlags::values);
  EXPECT_THROW(phi_m.get_face_value(0, phi_p), dealii::ExceptionBase);

  if (matrix_free.n_boundary_face_batches() > 0)
    {
      phi_m.reinit(matrix_free.n_inner_face_batches());
      phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);
      EXPECT_THROW(phi_m.get_face_value(0, phi_p), dealii::ExceptionBase);
    }
}
