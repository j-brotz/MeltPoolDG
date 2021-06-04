
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/utilities/utilityfunctions.hpp>

#include <iostream>

using namespace dealii;
using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int dim>
void
test(const unsigned int fe_degree, const unsigned int n_q_points)
{
  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(5);

  MappingQGeneric<dim> mapping(1);
  QGauss<1>            quadrature(n_q_points);
  FESystem<dim>        fe(FE_Q<dim>(fe_degree), dim);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> contraints;

  typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData additional_data;
  additional_data.mapping_update_flags = dealii::update_quadrature_points;

  MatrixFree<dim, double, VectorizedArray<double>> matrix_free;
  matrix_free.reinit(mapping, dof_handler, contraints, quadrature, additional_data);

  VectorType solution;
  matrix_free.initialize_dof_vector(solution);

  FECellIntegrator<dim, dim, double> fe_eval(matrix_free);

  MeltPoolDG::UtilityFunctions::fill_dof_vector_from_cell_operation_vec<dim, dim>(
    solution,
    matrix_free,
    0,
    0,
    fe_degree,
    n_q_points,
    [&](const auto cell, const auto q) -> Tensor<1, dim, VectorizedArray<double>> {
      fe_eval.reinit(cell);
      return fe_eval.quadrature_point(q);
    });

  if (false)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(dof_handler, solution, "solution");

      data_out.build_patches(mapping, n_q_points + 1);
      std::ofstream output("test" + std::to_string(fe_degree + n_q_points * 10) + ".vtu");
      data_out.write_vtu(output);
    }

  std::cout << solution.l2_norm() << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  for (unsigned int i = 1; i <= 5; ++i)                                  // fe_degree
    for (unsigned int j = std::max<unsigned int>(i, 2); j <= i + 2; ++j) // n_q_points_1D
      test<2>(i, j);

  return 0;
}
