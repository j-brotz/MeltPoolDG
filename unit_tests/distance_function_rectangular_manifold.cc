#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/utilities/distance_functions.hpp>

#include <iostream>

using namespace dealii;
using namespace MeltPoolDG;
using VectorType = LinearAlgebra::distributed::Vector<double>;

static constexpr double domain_size = 10.0;

template <int dim>
class Distance : public Function<dim>
{
public:
  Distance(const Point<dim> &bottom_left, const Point<dim> &top_right)
    : Function<dim>()
    , bottom_left(bottom_left)
    , top_right(top_right)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const
  {
    (void)component;

    return DistanceFunctions::rectangular_manifold<dim>(p, bottom_left, top_right);
  }

  const Point<dim> &bottom_left, top_right;
};

template <int dim>
void
test(const MPI_Comm &mpi_comm)
{
  const int degree = 1;

  Point<dim> bottom_left;
  Point<dim> top_right;

  for (unsigned int d = 0; d < dim; ++d)
    {
      bottom_left[d] = domain_size / 2.;
      top_right[d]   = 3. * domain_size / 4.;
    }

  // triangulation
  auto triangulation = parallel::shared::Triangulation<dim>(mpi_comm);

  GridGenerator::hyper_cube(triangulation, 0, domain_size);

  triangulation.refine_global(4);

  auto q_gauss = QGauss<dim>(degree + 1);
  auto fe_q    = FE_Q<dim>(degree);
  auto mapping = MappingFE<dim>(fe_q);

  DoFHandler<dim> dof_handler;
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;

  typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values | dealii::update_quadrature_points;

  MatrixFree<dim, double, VectorizedArray<double>> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, q_gauss, additional_data);

  dealii::ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  VectorType distance;
  matrix_free.initialize_dof_vector(distance, 0);

  VectorTools::interpolate(mapping, dof_handler, Distance<dim>(bottom_left, top_right), distance);

  // compute distance function of a rectangular manifold
  pcout << "dim= " << dim << " distance: " << distance.l2_norm() << std::endl;
  distance.update_ghost_values();

  // write paraview output
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(distance, "distance");

  data_out.build_patches(mapping);
  data_out.write_vtu_with_pvtu_record("./", "solution_" + std::to_string(dim), 0, mpi_comm);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_comm(MPI_COMM_WORLD);

  test<1>(mpi_comm);
  test<2>(mpi_comm);
  test<3>(mpi_comm);

  return 0;
}
