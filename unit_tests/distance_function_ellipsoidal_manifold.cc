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
  Distance(const Point<dim> &center, const Point<dim> &radii)
    : Function<dim>()
    , center(center)
    , radii(radii)
  {}

  double
  value(const Point<dim> &p, unsigned int) const override
  {
    return DistanceFunctions::ellipsoidal_manifold<dim>(p, center, radii, false);
  }

  const Point<dim> center, radii;
};

template <int dim>
void
test(const MPI_Comm &mpi_comm)
{
  const int degree = 1;

  Point<dim> center;
  Point<dim> radii;

  for (int d = 0; d < dim; ++d)
    {
      center[d] = domain_size / 3.;
      radii[d]  = domain_size / (3. + (double)d);
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

  dealii::ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  VectorType distance;
  IndexSet   locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  distance.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);

  const auto distance_function = Distance<dim>(center, radii);
  VectorTools::interpolate(mapping, dof_handler, distance_function, distance);

  // compute distance function of a rectangular manifold
  pcout << "dim= " << dim << " distance norm: " << distance.l2_norm() << std::endl;
  pcout << "distance at origin: " << distance_function.value(Point<dim>(), 0) << std::endl;
  pcout << "distance at center: " << distance_function.value(center, 0) << std::endl;
  pcout << "distance at [5, 3]: " << distance_function.value(Point<dim>(5., 3.), 0) << std::endl;
  pcout << "distance at [1, 8]: " << distance_function.value(Point<dim>(1., 8.), 0) << std::endl;
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

  test<2>(mpi_comm);

  return 0;
}
