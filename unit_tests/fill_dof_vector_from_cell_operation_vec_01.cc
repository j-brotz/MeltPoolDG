#include <deal.II/base/mpi.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <iostream>

using namespace dealii;
using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int dim, unsigned int n_components>
class MyTransformedFunction : public Function<dim>
{
public:
  MyTransformedFunction()
    : Function<dim>(n_components)
  {
    AssertThrow(n_components <= dim, dealii::ExcNotImplemented());
  }

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    if (dim == 2)
      {
        const auto &x = p[0];
        const auto &y = p[1];
        if (component == 0)
          return std::sin(dealii::numbers::PI * x) * std::cos(dealii::numbers::PI * y);
        else if (component == 1)
          return std::sin(dealii::numbers::PI * x) * std::sin(dealii::numbers::PI * y);
        else
          AssertThrow(false, ExcNotImplemented());
      }
    else if (dim == 3)
      {
        const auto &x = p[0];
        const auto &y = p[1];
        const auto &z = p[2];
        if (component == 0)
          return std::sin(dealii::numbers::PI * x) * std::cos(dealii::numbers::PI * y) *
                 std::cos(dealii::numbers::PI * z);
        else if (component == 1)
          return std::sin(dealii::numbers::PI * x) * std::sin(dealii::numbers::PI * y) *
                 std::cos(dealii::numbers::PI * z);
        else if (component == 2)
          return std::cos(dealii::numbers::PI * x) * std::cos(dealii::numbers::PI * y) *
                 std::sin(dealii::numbers::PI * z);
        else
          AssertThrow(false, ExcNotImplemented());
      }
  }
};

template <int dim, unsigned int n_components>
void
test(const unsigned int fe_degree,
     const unsigned int n_q_points,
     bool               do_local_refinement,
     TableHandler      &table)
{
  MyTransformedFunction<dim, n_components> my_func;

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
        MeltPoolDG::FECellIntegrator<dim, 1, double> fe_eval(matrix_free);
        fe_eval.reinit(cell);
        return MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points<dim, double>(
          my_func, fe_eval.quadrature_point(q), 0);
      });
  else
    MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, n_components>(
      solution,
      matrix_free,
      0,
      0,
      [&](const auto cell, const auto q) -> Tensor<1, n_components, VectorizedArray<double>> {
        MeltPoolDG::FECellIntegrator<dim, n_components, double> fe_eval(matrix_free);
        fe_eval.reinit(cell);
        return MeltPoolDG::VectorTools::
          evaluate_function_at_vectorized_points<dim, double, n_components>(
            my_func, fe_eval.quadrature_point(q));
      });

  constraints.distribute(solution);

  if (false)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.add_data_vector(dof_handler, solution, "solution");

      VectorType analytical_solution;
      matrix_free.initialize_dof_vector(analytical_solution);
      dealii::VectorTools::interpolate(mapping, dof_handler, my_func, analytical_solution);
      data_out.add_data_vector(dof_handler, analytical_solution, "analytical_solution");

      data_out.build_patches(mapping, n_q_points + 1);
      std::ofstream output("test" +
                           std::to_string(fe_degree + n_q_points * 10 + n_components * 100 +
                                          (int)do_local_refinement * 1000) +
                           ".vtu");
      data_out.write_vtu(output);
    }

  // compute L2Norm of level_set
  dealii::Vector<double> difference_per_cell(triangulation.n_active_cells());
  dealii::VectorTools::integrate_difference<dim>(mapping,
                                                 dof_handler,
                                                 solution,
                                                 my_func,
                                                 difference_per_cell,
                                                 QGauss<dim>(n_q_points),
                                                 dealii::VectorTools::L2_norm);

  const double error = dealii::VectorTools::compute_global_error(triangulation,
                                                                 difference_per_cell,
                                                                 dealii::VectorTools::L2_norm);
  table.add_value("degree", fe_degree);
  table.add_value("n_comp", n_components);
  table.add_value("L2_error", error);
  table.add_value("n_q_points_1D", n_q_points);
  table.add_value("do_local_refinement", do_local_refinement);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  for (unsigned int deg = 1; deg <= 5; ++deg)
    { // fe_degree

      TableHandler table;
      table.declare_column("n_comp");
      table.declare_column("do_local_refinement");
      table.declare_column("degree");
      table.declare_column("n_q_points_1D");
      table.declare_column("L2_error");
      table.set_scientific("L2_error", true);

      unsigned int n_q_points_1D = deg + 1;
      test<2, 1 /*n_components*/>(deg, n_q_points_1D, false, table);
      test<2, 2>(deg, n_q_points_1D, false, table);
      test<2, 1>(deg, n_q_points_1D, true, table);
      test<2, 2>(deg, n_q_points_1D, true, table);
      table.write_text(std::cout);
    }

  return 0;
}
