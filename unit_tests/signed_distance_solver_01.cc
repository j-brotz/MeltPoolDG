// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

// Deal.II includes
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/level_set/signed_distance_solver.hpp>

#include <iostream>

using namespace dealii;

template <int dim, typename VectorType>
void
output_vtu(const DoFHandler<dim> &dof_handler,
           const VectorType      &level_set,
           const VectorType      &signed_distance,
           const std::string     &filename)
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(level_set, "level_set");
  data_out.add_data_vector(signed_distance, "signed_distance");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", filename, 0, MPI_COMM_WORLD, 2);
}

template <int dim>
void
test()
{
  using namespace MeltPoolDG;

  /* This test checks the SignedDistanceSolver class from within
  an arbitrary structure that replicates the usual physical solver structure,
  denoted by the background prefix. This test performs verification of the
  signed distance computations for a sphere.
  */
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));

  // Triangulation (as a shared_ptr to reproduce an arbitrary background solver
  // architecture)
  std::shared_ptr<parallel::DistributedTriangulationBase<dim>> background_triangulation =
    std::make_shared<parallel::distributed::Triangulation<dim>>(
      mpi_communicator,
      typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement |
                                                 Triangulation<dim>::smoothing_on_coarsening));

  Point<dim> p_0 = Point<dim>();
  Point<dim> p_1 = Point<dim>();
  for (int n = 0; n < dim; n++)
    {
      p_0[n] = 0.0;
      p_1[n] = 1.0;
    }
  GridGenerator::hyper_rectangle(*background_triangulation, p_0, p_1);
  background_triangulation->refine_global(3);

  const double sphere_radius = 0.25;
  Point<dim>   sphere_center = Point<dim>();
  for (int n = 0; n < dim; n++)
    {
      sphere_center[n] = 0.5;
    }

  const auto level_set_function =
    Functions::SignedDistance::Sphere<dim>(sphere_center, sphere_radius);

  // for (unsigned int refinement_cycle = 0; refinement_cycle < 2;
  //++refinement_cycle)
  //{
  // for (const auto &cell : background_triangulation->active_cell_iterators())
  // if (cell->is_locally_owned())
  //{
  // bool has_negative = false;
  // bool has_positive = false;

  // for (const unsigned int v : cell->vertex_indices())
  //{
  // const double phi = level_set_function.value(cell->vertex(v));

  // has_negative |= (phi < 0.0);
  // has_positive |= (phi > 0.0);
  //}

  // if (has_negative && has_positive)
  // cell->set_refine_flag();
  //}

  // background_triangulation->execute_coarsening_and_refinement();
  //}


  // Discretize the domain as in an arbitrary background solver
  std::shared_ptr<FiniteElement<dim>> background_fe = std::make_shared<FE_Q<dim>>(1);
  std::shared_ptr<MappingQ<dim>>      background_mapping =
    std::make_shared<MappingQ<dim>>(background_fe->degree);

  DoFHandler<dim> background_dof_handler(*background_triangulation);
  background_dof_handler.distribute_dofs(*background_fe);

  IndexSet background_locally_owned_dofs = background_dof_handler.locally_owned_dofs();
  IndexSet background_locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(background_dof_handler);

  // Set the background level-set field of the sphere
  LinearAlgebra::distributed::Vector<double> background_level_set(background_locally_owned_dofs,
                                                                  background_locally_relevant_dofs,
                                                                  mpi_communicator);

  VectorTools::interpolate(*background_mapping,
                           background_dof_handler,
                           Functions::SignedDistance::Sphere<dim>(sphere_center, sphere_radius),
                           background_level_set);

  output_vtu(background_dof_handler,
             background_level_set,
             background_level_set,
             "initial_" + std::to_string(dim) + "d");

  // Instanciate the SignedDistanceSolver as it would as a member of the
  // background solver class
  double max_reinitialization_distance = 1.0;

  std::shared_ptr<LevelSet::SignedDistanceSolver<dim, LinearAlgebra::distributed::Vector<double>>>
    signed_distance_solver = std::make_shared<
      LevelSet::SignedDistanceSolver<dim, LinearAlgebra::distributed::Vector<double>>>(
      background_dof_handler,
      max_reinitialization_distance,
      0.0,
      1.0,
      LevelSet::Verbosity::verbose);

  signed_distance_solver->setup_dofs();

  signed_distance_solver->set_level_set_from_background_mesh(background_dof_handler,
                                                             background_level_set);

  // Solve the signed_distance field.
  signed_distance_solver->solve();

  output_vtu(background_dof_handler,
             background_level_set,
             signed_distance_solver->get_signed_distance(),
             "signed_distance_" + std::to_string(dim) + "d");

  // Compute the L2 norm of the error.
  Vector<float> error_per_cell(background_triangulation->n_active_cells());

  const auto signed_distance = signed_distance_solver->get_signed_distance();
  signed_distance.update_ghost_values();


  // Interpolate the signed_distance solution from the SignedDistanceSolver
  // DoFHandler to the main solver DoFHandler.
  LinearAlgebra::distributed::Vector<double> background_signed_distance(
    background_locally_owned_dofs, background_locally_relevant_dofs, mpi_communicator);

  LinearAlgebra::distributed::Vector<double> tmp_background_signed_distance(
    background_locally_owned_dofs, mpi_communicator);

  FETools::interpolate(background_dof_handler,
                       signed_distance_solver->get_signed_distance(),
                       background_dof_handler,
                       tmp_background_signed_distance);

  background_signed_distance = tmp_background_signed_distance;


  VectorTools::integrate_difference(*background_mapping,
                                    background_dof_handler,
                                    background_signed_distance,
                                    Functions::SignedDistance::Sphere<dim>(sphere_center,
                                                                           sphere_radius),
                                    error_per_cell,
                                    QGauss<dim>(background_fe->degree + 1),
                                    VectorTools::L2_norm);

  const double error_L2 = VectorTools::compute_global_error(*background_triangulation,
                                                            error_per_cell,
                                                            VectorTools::L2_norm);

  if (this_mpi_process == 0)
    std::cout << "The L2 norm of the signed distance error in " << dim << "D is: " << error_L2
              << std::endl;
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      test<2>();
      test<3>();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  return 0;
}
