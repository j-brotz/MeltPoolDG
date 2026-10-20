/**
 * @file subcell_matrix_free_01.cpp
 * @brief Unit tests for the classes SubcellMatrixFree, SubcellShapeInfo and SubcellIndexInfo.
 */

#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/subcell_matrix_free.templates.hpp>

#include <array>
#include <utility>
#include <vector>

#include "../test_utils/test_functions.hpp"

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
 * Test fixture setting up a DG discretization on a uniform, distributed mesh of the unit cube,
 * with all faces of the locally owned cells held by the MatrixFree object.
 */
template <typename Config>
class SubcellMatrixFreeTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = Config::dim;
  static constexpr unsigned int degree       = Config::degree;
  static constexpr int          n_components = Config::n_components;

  static constexpr int n_cells_1d = dim == 2 ? 6 : 4;

  SubcellMatrixFreeTest()
    : triangulation(MPI_COMM_WORLD)
    , scratch_data(MPI_COMM_WORLD, 0, true)
    , fe_data(FiniteElementType::FE_DGQ, degree)
  {}

  /**
   * Create the mesh and the matrix-free data structures.
   */
  void
  setup()
  {
    std::vector<unsigned int> repetitions(dim, n_cells_1d);
    dealii::Point<dim>        unit_point;
    for (unsigned int d = 0; d < dim; ++d)
      unit_point[d] = 1.0;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, repetitions, dealii::Point<dim>(), unit_point, /*colorize=*/true);

    dof_handler.reinit(triangulation);
    FiniteElementUtils::distribute_dofs<dim, n_components>(fe_data, dof_handler);
    constraints.close();

    scratch_data.set_mapping(FiniteElementUtils::create_mapping<dim>(fe_data));
    scratch_data.attach_quadrature(FiniteElementUtils::create_quadrature<dim>(fe_data));
    scratch_data.attach_dof_handler(dof_handler);
    scratch_data.attach_constraint_matrix(constraints);
    scratch_data.create_partitioning();
    scratch_data.build(true, true, false, false, true);
  }

  dealii::parallel::distributed::Triangulation<dim> triangulation;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::AffineConstraints<double>                 constraints;
  ScratchData<dim, dim, double>                     scratch_data;
  FiniteElementData                                 fe_data;
};

using TestConfigs = ::testing::
  Types<TestConfiguration<2, 1, 2>, TestConfiguration<2, 3, 3>, TestConfiguration<3, 2, 2>>;

TYPED_TEST_SUITE(SubcellMatrixFreeTest, TestConfigs);

/**
 * The subcells given by SubcellIndexInfo::face_to_cell_subcell_indices must be adjacent to the face
 * and in the order of the quadrature points of FEFaceEvaluation. For all inner faces, the test
 * compares the location of each such subcell, taken from FEEvaluation on the cells of both sides,
 * with the corresponding face quadrature point in the tangential directions.
 */
TYPED_TEST(SubcellMatrixFreeTest, FaceSubcellsMatchFaceQuadraturePoints)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup();

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);
  const SubcellShapeInfo<dim, double> &shape_info = subcell_matrix_free.get_subcell_shape_info();
  const unsigned int                   n          = shape_info.cell_index_info.n_subcells_1d;

  dealii::FEEvaluation<dim, -1, 0, n_components, double>     phi_cell(matrix_free, 0, 0);
  dealii::FEFaceEvaluation<dim, -1, 0, n_components, double> phi_face(matrix_free, true, 0, 0);

  for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
    for (const bool interior : {true, false})
      {
        const dealii::internal::MatrixFreeFunctions::FaceToCellTopology<n_lanes> &face_info =
          matrix_free.get_face_info(face);
        const unsigned int face_no =
          interior ? face_info.interior_face_no : face_info.exterior_face_no;
        const unsigned int normal_direction = face_no / 2;

        phi_face.reinit(face);
        phi_cell.reinit(interior ? face_info.cells_interior : face_info.cells_exterior);

        ASSERT_EQ(shape_info.cell_index_info.face_to_cell_subcell_indices[face_no].size(),
                  phi_face.n_q_points);
        for (unsigned int q = 0; q < phi_face.n_q_points; ++q)
          {
            const unsigned int subcell =
              shape_info.cell_index_info.face_to_cell_subcell_indices[face_no][q];

            // the subcell lies in the layer adjacent to the face
            EXPECT_EQ((subcell / dealii::Utilities::pow(n, normal_direction)) % n,
                      (face_no % 2) * (n - 1))
              << "face_no " << face_no << ", q " << q;

            const dealii::Point<dim, VectorizedArrayType> face_point = phi_face.quadrature_point(q);
            const dealii::Point<dim, VectorizedArrayType> subcell_point =
              phi_cell.quadrature_point(subcell);
            for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face);
                 ++lane)
              for (unsigned int d = 0; d < dim; ++d)
                {
                  if (d != normal_direction)
                    {
                      EXPECT_NEAR(subcell_point[d][lane], face_point[d][lane], tolerance)
                        << "face batch " << face << ", interior " << interior << ", q " << q
                        << ", lane " << lane << ", direction " << d;
                    }
                }
          }
      }
}

/**
 * The intra cell subcell faces of SubcellIndexInfo must be numbered as in FESubcellEvaluation,
 * i.e. with the same interior and exterior subcells and the same normal directions.
 */
TYPED_TEST(SubcellMatrixFreeTest, IntraCellFacesMatchFESubcellEvaluation)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup();

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);
  const SubcellShapeInfo<dim, double> &shape_info = subcell_matrix_free.get_subcell_shape_info();

  FESubcellEvaluation<dim, n_components, double> phi(subcell_matrix_free);
  phi.reinit(0);

  ASSERT_EQ(shape_info.cell_index_info.intra_cell_face_to_subcells.size(),
            shape_info.cell_index_info.n_intra_cell_faces);
  ASSERT_EQ(shape_info.cell_index_info.intra_cell_face_directions.size(),
            shape_info.cell_index_info.n_intra_cell_faces);
  unsigned int n_faces = 0;
  for (const unsigned int f : phi.subcell_face_indices())
    {
      const std::pair<unsigned int, unsigned int> subcells =
        phi.subcell_face_interior_exterior_indices(f);
      EXPECT_EQ(shape_info.cell_index_info.intra_cell_face_to_subcells[f][0], subcells.first)
        << "face " << f;
      EXPECT_EQ(shape_info.cell_index_info.intra_cell_face_to_subcells[f][1], subcells.second)
        << "face " << f;

      const dealii::Tensor<1, dim, VectorizedArrayType> normal = phi.subcell_face_normal_vector(f);
      EXPECT_EQ(normal[shape_info.cell_index_info.intra_cell_face_directions[f]][0], 1.)
        << "face " << f;
      ++n_faces;
    }
  EXPECT_EQ(n_faces, shape_info.cell_index_info.n_intra_cell_faces);
}

/**
 * With hold_all_faces_to_owned_cells, the face neighbors of all locally owned cells are held by
 * the MatrixFree object, so each of them is either held or at the domain boundary, and none is an
 * extra cell.
 */
TYPED_TEST(SubcellMatrixFreeTest, NeighborsOfOwnedCellsAreHeldOrAtBoundary)
{
  constexpr int dim = TestFixture::dim;
  this->setup();

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
    for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
      {
        const typename SubcellMatrixFree<dim, double>::FaceNeighbors &neighbors =
          subcell_matrix_free.get_face_neighbors(cell_batch, face_no);

        bool all_held = true;
        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
             ++lane)
          {
            const typename dealii::DoFHandler<dim>::cell_iterator cell =
              matrix_free.get_cell_iterator(cell_batch, lane);

            EXPECT_EQ(neighbors.extra_cell_ids[lane], dealii::numbers::invalid_unsigned_int);
            if (cell->at_boundary(face_no))
              {
                EXPECT_EQ(neighbors.cell_ids[lane], dealii::numbers::invalid_unsigned_int);
                EXPECT_EQ(neighbors.boundary_ids[lane], cell->face(face_no)->boundary_id());
                EXPECT_EQ(neighbors.extents[lane], 0.);
                all_held = false;
              }
            else
              {
                ASSERT_NE(neighbors.cell_ids[lane], dealii::numbers::invalid_unsigned_int);
                EXPECT_EQ(matrix_free.get_cell_iterator(neighbors.cell_ids[lane] / n_lanes,
                                                        neighbors.cell_ids[lane] % n_lanes),
                          cell->neighbor(face_no));
                EXPECT_EQ(neighbors.boundary_ids[lane], dealii::numbers::invalid_boundary_id);
                EXPECT_NEAR(neighbors.extents[lane], 1. / TestFixture::n_cells_1d, tolerance);
              }
          }
        EXPECT_EQ(neighbors.all_held, all_held)
          << "cell batch " << cell_batch << ", face_no " << face_no;
      }
}

/**
 * The subcell stencil on a face reaches over the lateral faces of both adjacent cells. For all
 * inner faces, the lateral neighbors of the cells on both sides must therefore be available, i.e.
 * either held, extra or at the domain boundary, also if the cell is a ghost cell. With more than
 * two MPI processes, some of those neighbors are extra cells at corners of the partition boundary.
 */
TYPED_TEST(SubcellMatrixFreeTest, LateralNeighborsOfFaceCellsAreAvailable)
{
  constexpr int dim = TestFixture::dim;
  this->setup();

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  unsigned int n_extra_neighbors = 0;
  for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
    for (const bool interior : {true, false})
      {
        const dealii::internal::MatrixFreeFunctions::FaceToCellTopology<n_lanes> &face_info =
          matrix_free.get_face_info(face);
        const std::array<unsigned int, n_lanes> &cell_ids =
          interior ? face_info.cells_interior : face_info.cells_exterior;

        for (unsigned int lateral_face_no = 0; lateral_face_no < 2 * dim; ++lateral_face_no)
          {
            if (lateral_face_no / 2 == face_info.interior_face_no / 2)
              continue;

            const typename SubcellMatrixFree<dim, double>::FaceNeighbors neighbors =
              subcell_matrix_free.get_face_neighbors(cell_ids, lateral_face_no);

            for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face);
                 ++lane)
              {
                const typename dealii::DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(cell_ids[lane] / n_lanes, cell_ids[lane] % n_lanes);

                if (cell->at_boundary(lateral_face_no))
                  {
                    EXPECT_EQ(neighbors.cell_ids[lane], dealii::numbers::invalid_unsigned_int);
                    EXPECT_EQ(neighbors.extra_cell_ids[lane],
                              dealii::numbers::invalid_unsigned_int);
                    EXPECT_EQ(neighbors.boundary_ids[lane],
                              cell->face(lateral_face_no)->boundary_id());
                    EXPECT_FALSE(neighbors.all_held);
                    continue;
                  }

                EXPECT_EQ(neighbors.boundary_ids[lane], dealii::numbers::invalid_boundary_id);
                EXPECT_NEAR(neighbors.extents[lane], 1. / TestFixture::n_cells_1d, tolerance);
                if (neighbors.cell_ids[lane] != dealii::numbers::invalid_unsigned_int)
                  {
                    EXPECT_EQ(neighbors.extra_cell_ids[lane],
                              dealii::numbers::invalid_unsigned_int);
                    EXPECT_EQ(matrix_free.get_cell_iterator(neighbors.cell_ids[lane] / n_lanes,
                                                            neighbors.cell_ids[lane] % n_lanes),
                              cell->neighbor(lateral_face_no));
                  }
                else
                  {
                    EXPECT_FALSE(cell->is_locally_owned());
                    EXPECT_LT(neighbors.extra_cell_ids[lane], subcell_matrix_free.n_extra_cells())
                      << "face batch " << face << ", interior " << interior << ", lane " << lane
                      << ", lateral face_no " << lateral_face_no;
                    EXPECT_FALSE(neighbors.all_held);
                    ++n_extra_neighbors;
                  }
              }
          }
      }

  // With two processes, the partition boundary is a plane without corners, so extra cells only
  // occur with more processes.
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 2)
    {
      EXPECT_GT(dealii::Utilities::MPI::sum(n_extra_neighbors, MPI_COMM_WORLD), 0u);
    }
}

/**
 * The DoF values of the extra cells, exchanged with update_ghost_values() and read with
 * read_extra_cell_dof_values(), must be in the order of FEEvaluation. The test interpolates a
 * linear function, which the DG space represents exactly, evaluates the DoF values of each extra
 * cell with FEEvaluation and compares the result with the function at the quadrature points of the
 * extra cell. The ghost values of the vector itself are not updated, since only its locally owned
 * values are used.
 */
TYPED_TEST(SubcellMatrixFreeTest, ExtraCellDofValuesMatchInterpolatedFunction)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup();

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const TestUtils::LinearFunction<dim, n_components> function;
  VectorType                                         solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  dealii::VectorTools::interpolate(this->scratch_data.get_mapping(),
                                   this->dof_handler,
                                   function,
                                   solution);

  EXPECT_FALSE(subcell_matrix_free.has_ghost_values_of(solution));
  subcell_matrix_free.update_ghost_values(solution);
  EXPECT_TRUE(subcell_matrix_free.has_ghost_values_of(solution));

  dealii::FEEvaluation<dim, -1, 0, n_components, double> phi(matrix_free, 0, 0);
  const dealii::Quadrature<1> &quadrature_1d = matrix_free.get_shape_info(0, 0).data[0].quadrature;
  dealii::AlignedVector<VectorizedArrayType> dof_values(phi.dofs_per_cell);

  unsigned int n_extra_neighbors = 0;
  for (unsigned int cell_batch = matrix_free.n_cell_batches();
       cell_batch < matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
       ++cell_batch)
    for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
      {
        const typename SubcellMatrixFree<dim, double>::FaceNeighbors &neighbors =
          subcell_matrix_free.get_face_neighbors(cell_batch, face_no);

        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
             ++lane)
          {
            if (neighbors.extra_cell_ids[lane] == dealii::numbers::invalid_unsigned_int)
              continue;

            const typename dealii::DoFHandler<dim>::cell_iterator extra_cell =
              matrix_free.get_cell_iterator(cell_batch, lane)->neighbor(face_no);
            EXPECT_TRUE(extra_cell->is_ghost());
            EXPECT_EQ(matrix_free.get_matrix_free_cell_index(extra_cell),
                      dealii::numbers::invalid_unsigned_int);

            // only lane 0 is evaluated, the cell batch only provides the shape data
            subcell_matrix_free.read_extra_cell_dof_values(neighbors.extra_cell_ids[lane],
                                                           0,
                                                           dof_values.data());
            phi.reinit(0);
            phi.evaluate(dof_values.data(), dealii::EvaluationFlags::values);

            for (const unsigned int q : phi.quadrature_point_indices())
              {
                dealii::Point<dim> unit_point;
                unsigned int       remaining_index = q;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    unit_point[d] = quadrature_1d.point(remaining_index % quadrature_1d.size())[0];
                    remaining_index /= quadrature_1d.size();
                  }
                const dealii::Point<dim> point =
                  this->scratch_data.get_mapping().transform_unit_to_real_cell(extra_cell,
                                                                               unit_point);

                const typename dealii::FEEvaluation<dim, -1, 0, n_components, double>::value_type
                  value = phi.get_value(q);
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    double computed_value;
                    if constexpr (n_components == 1)
                      computed_value = value[0];
                    else
                      computed_value = value[c][0];
                    EXPECT_NEAR(computed_value, function.value(point, c), tolerance)
                      << "extra cell " << neighbors.extra_cell_ids[lane] << ", q " << q
                      << ", component " << c;
                  }
              }
            ++n_extra_neighbors;
          }
      }

  // With two processes, the partition boundary is a plane without corners, so extra cells only
  // occur with more processes.
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 2)
    {
      EXPECT_GT(dealii::Utilities::MPI::sum(n_extra_neighbors, MPI_COMM_WORLD), 0u);
    }
}
