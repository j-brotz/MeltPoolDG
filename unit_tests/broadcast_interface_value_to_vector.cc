#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <iostream>

using namespace dealii;
using namespace MeltPoolDG;

using VectorType      = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

/**
 * Initial fields
 */
template <int dim, unsigned int n_components = 1>
class InitializeTemperature : public Function<dim>
{
public:
  InitializeTemperature()
    : Function<dim>(n_components)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const
  {
    const double T_max = 500;
    const double T_min = 0;
    return T_max - (T_max - T_min) / 1. * p[component];
  }
};

template <int dim>
class InitializeDistance : public Function<dim>
{
public:
  InitializeDistance()
    : Function<dim>()
  {}
  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/) const
  {
    // set radius of bubble to 0.5, slightly shifted away from the center
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
      center[d] = 0.5;

    return DistanceFunctions::spherical_manifold<dim>(p, center, 0.25);
  }
};

template <int n_components, int n_refinements, int n_iter>
void
run_test()
{
  MPI_Comm                   mpi_comm(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  pcout << "--------------------------------------------------------" << std::endl;
  pcout << " START TEST: n_iter=" << n_iter << std::endl;
  pcout << "--------------------------------------------------------" << std::endl;

  Timer     timer_total(mpi_comm);
  const int dim         = 2;
  const int degree      = 1;
  const int degree_temp = 2;

  parallel::distributed::Triangulation<dim> triangulation(mpi_comm);

  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refinements);

  /*
   * Output mesh per processor
   */
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(triangulation, "grid");

  MappingQGeneric<dim> mapping(degree);
  FE_Q<dim>            fe(degree);
  DoFHandler<dim>      dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  /*
   * setup level set related vectors
   */
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  VectorType solution_distance;
  solution_distance.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  /*
   * compute normals
   */
  BlockVectorType solution_normal(dim);
  for (unsigned int d = 0; d < dim; ++d)
    solution_normal.block(d).reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  {
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
      center[d] = 0.5;

    FEValues<dim> normal_eval(mapping,
                              fe,
                              Quadrature<dim>(fe.get_unit_support_points()),
                              update_quadrature_points);

    const unsigned int                   dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            normal_eval.reinit(cell);

            for (const auto q : normal_eval.quadrature_point_indices())
              {
                Point<dim> normal_vec;
                normal_vec = center - normal_eval.quadrature_point(q);
                normal_vec /= normal_vec.norm();

                // auto idx = fe_dim.system_to_component_index(q);
                for (unsigned int d = 0; d < dim; ++d)
                  solution_normal.block(d)[local_dof_indices[q]] = normal_vec[d];
              }
          }
      }
  }
  /*
   * interpolate distance onto quadrature points
   */
  VectorTools::interpolate(mapping, dof_handler, InitializeDistance<dim>(), solution_distance);

  /*
   * setup temperature field
   */

  DoFHandler<dim> dof_handler_temp(triangulation);
  if constexpr (n_components == 1)
    {
      FE_Q<dim> fe_temp(degree_temp);
      dof_handler_temp.distribute_dofs(fe_temp);
    }
  else
    {
      FESystem<dim> fe_temp(FE_Q<dim>(degree_temp), n_components);
      dof_handler_temp.distribute_dofs(fe_temp);
    }

  VectorType solution_temp, solution_temp_interface;
  locally_owned_dofs = dof_handler_temp.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_temp, locally_relevant_dofs);

  solution_temp.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  solution_temp_interface.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  /*
   * interpolate temperature onto quadrature points
   */
  VectorTools::interpolate(mapping,
                           dof_handler_temp,
                           InitializeTemperature<dim, n_components>(),
                           solution_temp);
  /*
   * ------------------------------------------------------------------------------------
   * ------------------------------------------------------------------------------------
   *      START OF ACTUAL ALGORITHM
   * ------------------------------------------------------------------------------------
   * ------------------------------------------------------------------------------------
   */
  Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(1e-6 /*tolerance*/,
                                                                          true /*unique mapping*/);

  const double min_cell_size = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(dim);

  timer_total.start();
  LevelSet::Tools::broadcast_interface_value_to_vector<dim, n_components>(
    mapping,
    dof_handler,
    dof_handler_temp,
    solution_distance,
    solution_normal,
    solution_temp,
    solution_temp_interface,
    min_cell_size,
    remote_point_evaluation,
    n_iter,
    1e-5,   // rel_tol_distance,
    0.125); // distance interval for projection
  timer_total.stop();
  /*
   * ------------------------------------------------------------------------------------
   * ------------------------------------------------------------------------------------
   *      END OF ACTUAL ALGORITHM
   * ------------------------------------------------------------------------------------
   * ------------------------------------------------------------------------------------
   */

  pcout << "norm of interface vals: " << solution_temp_interface.l2_norm() << std::endl;
  /*
   * debug
   */
  if (false)
    {
      /*
       * add output vectors
       */
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(dof_handler, solution_distance, "solution_distance");
      for (unsigned int d = 0; d < dim; ++d)
        data_out.add_data_vector(dof_handler,
                                 solution_normal.block(d),
                                 "normal_" + std::to_string(d));

      if constexpr (n_components > 1)
        {
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            vector_component_interpretation(
              n_components, DataComponentInterpretation::component_is_part_of_vector);

          data_out.add_data_vector(dof_handler_temp,
                                   solution_temp_interface,
                                   std::vector<std::string>(n_components,
                                                            "solution_temp_interface"),
                                   vector_component_interpretation);

          data_out.add_data_vector(dof_handler_temp,
                                   solution_temp,
                                   std::vector<std::string>(n_components, "solution_temp"),
                                   vector_component_interpretation);
        }
      else
        {
          AssertThrow(n_components == dim, ExcNotImplemented());
          data_out.add_data_vector(dof_handler_temp,
                                   solution_temp_interface,
                                   "solution_temp_interface");
          data_out.add_data_vector(dof_handler_temp, solution_temp, "solution_temp");
        }


      pcout << "Elapsed wall time " << timer_total.wall_time() << std::endl;
      data_out.build_patches(mapping);
      std::string output = "n_components_" + std::to_string(n_components) + "n_iter_" +
                           std::to_string(n_iter) + "_output.vtu";
      data_out.write_vtu_in_parallel(output, mpi_comm);
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  run_test<1, 6, 1>();
  run_test<1, 6, 16>();
  run_test<2, 6, 5>();

  return 0;
}
