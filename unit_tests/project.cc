#include <deal.II/base/function_signed_distance.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/utilities/vector_tools.hpp>

#include <iostream>

using namespace dealii;
using namespace MeltPoolDG;

using VectorType = LinearAlgebra::distributed::Vector<double>;


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const int dim    = 2;
  const int degree = 2;

  Triangulation<dim> triangulation;

  GridGenerator::hyper_cube(triangulation, -1.5, 1.5);

  triangulation.refine_global(5);

  QGauss<dim>    quad(degree + 1);
  FE_Q<dim>      fe_q(degree);
  MappingQ1<dim> mapping;

  DoFHandler<dim> dof_handler;
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;

  VectorType rhs(dof_handler.n_dofs());
  VectorType rhs_projected(dof_handler.n_dofs());

  // create righ-hand-side vector
  dealii::VectorTools::create_right_hand_side(
    mapping, dof_handler, quad, Functions::SignedDistance::Sphere<dim>(), rhs);

  // project vector
  MeltPoolDG::VectorTools::project_vector<1>(
    mapping, dof_handler, constraints, quad, rhs, rhs_projected);

  Vector<double> difference_per_cell(triangulation.n_active_cells());

  dealii::VectorTools::integrate_difference(mapping,
                                            dof_handler,
                                            rhs_projected,
                                            Functions::SignedDistance::Sphere<dim>(),
                                            difference_per_cell,
                                            quad,
                                            dealii::VectorTools::L2_norm);


  if (dealii::VectorTools::compute_global_error(triangulation,
                                                difference_per_cell,
                                                dealii::VectorTools::L2_norm) < 1e-4)
    std::cout << "OK!" << std::endl;

#if false
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(rhs, "rhs");
  data_out.add_data_vector(rhs_projected, "rhs_projected");
  data_out.build_patches(2);
  data_out.write_vtu_with_pvtu_record("./", "result", 0, triangulation.get_communicator());
#endif

  return 0;
}
