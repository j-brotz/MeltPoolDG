#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/utilities/vector_tools.hpp>

#include <iostream>

using namespace dealii;
using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_q_points,
     const unsigned int n_components,
     bool               do_local_refinement)
{
  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(5);

  if (do_local_refinement)
    {
      const unsigned int n_local_refinement = 2;

      Point<dim> bl;
      Point<dim> tr;
      for (unsigned int d = 0; d < dim; ++d)
        {
          bl[d] = 0.5;
          tr[d] = 0.75;
        }

      const auto refinement_region = BoundingBox<dim>({bl, tr});

      for (unsigned int j = 0; j < n_local_refinement; ++j)
        {
          for (auto &cell : triangulation.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                    if (refinement_region.point_inside(cell->vertex(i)))
                      {
                        cell->set_refine_flag();
                        break;
                      }
                }
            }
          triangulation.execute_coarsening_and_refinement();
        }
    }

  MappingQGeneric<dim> mapping(1);
  QGauss<1>            quadrature(n_q_points);
  FESystem<dim>        fe(FE_Q<dim>(fe_degree), n_components);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;

  if (do_local_refinement)
    {
      constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    }

  constraints.close();

  typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData additional_data;
  additional_data.mapping_update_flags = dealii::update_quadrature_points;

  MatrixFree<dim, double, VectorizedArray<double>> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quadrature, additional_data);

  VectorType solution;
  matrix_free.initialize_dof_vector(solution);

  if (n_components == 1)
    MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
      solution, matrix_free, 0, 0, [&](const auto cell, const auto q) -> VectorizedArray<double> {
        FECellIntegrator<dim, 1, double> fe_eval(matrix_free);
        fe_eval.reinit(cell);
        return fe_eval.quadrature_point(q)[0];
      });
  else
    MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, dim>(
      solution,
      matrix_free,
      0,
      0,
      [&](const auto cell, const auto q) -> Tensor<1, dim, VectorizedArray<double>> {
        FECellIntegrator<dim, dim, double> fe_eval(matrix_free);
        fe_eval.reinit(cell);
        return fe_eval.quadrature_point(q);
      });

  constraints.distribute(solution);

  if (true)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(dof_handler, solution, "solution");

      data_out.build_patches(mapping, n_q_points + 1);
      std::ofstream output("test" +
                           std::to_string(fe_degree + n_q_points * 10 + n_components * 100 +
                                          (int)do_local_refinement * 1000) +
                           ".vtu");
      data_out.write_vtu(output);
    }

  std::cout << solution.l2_norm() << " ";
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  for (unsigned int i = 1; i <= 5; ++i)                                  // fe_degree
    for (unsigned int j = std::max<unsigned int>(i, 2); j <= i + 2; ++j) // n_q_points_1D
      {
        test<2>(i, j, 1, false);
        test<2>(i, j, 2, false);
        test<2>(i, j, 1, true);
        test<2>(i, j, 2, true);
        std::cout << std::endl;
      }

  return 0;
}
