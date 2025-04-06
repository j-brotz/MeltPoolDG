#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/cut/cut_norm.hpp>
#include <meltpooldg/cut/util.hpp>

#include <iostream>
#include <memory>
#include <vector>

using namespace dealii;
using namespace MeltPoolDG::CutUtil;



// interpolating this function on the solution vector makes the first phase's
// values 1.0 and the second phase's values 2.0
template <int dim>
class StepFunction : public Function<dim>
{
public:
  StepFunction()
    : Function<dim>(2)
  {}

  double
  value(const Point<dim> &, const unsigned int component) const override
  {
    return component == 0 ? 1.0 : 2.0;
  }
};


template <int dim>
void
test()
{
  using number = double;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(tria, 9, -1, 1);

  const bool            is_two_phase = true;
  const unsigned int    degree       = 1;
  const FE_Q<dim>       fe_q(degree);
  const FE_Nothing<dim> fe_n;
  hp::FECollection<dim> fe_collection;
  fe_collection.push_back(FESystem<dim, dim>(fe_q, 1, fe_n, 1)); // inside
  fe_collection.push_back(FESystem<dim, dim>(fe_q, 1, fe_q, 1)); // intersected
  fe_collection.push_back(FESystem<dim, dim>(fe_n, 1, fe_q, 1)); // outside
  MappingQ<dim> mapping(degree);

  DoFHandler<dim> dof_handler_cut(tria);
  DoFHandler<dim> dof_handler_ls(tria);

  dof_handler_ls.distribute_dofs(fe_q);
  LinearAlgebra::distributed::Vector<number> level_set;
  level_set.reinit(dof_handler_ls.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(dof_handler_ls),
                   tria.get_communicator());
  const Functions::SignedDistance::Plane<dim> signed_distance(Point<dim>(),
                                                              Point<dim>::unit_vector(0));
  VectorTools::interpolate(dof_handler_ls, signed_distance, level_set);
  level_set.update_ghost_values();
  NonMatching::MeshClassifier<dim> mesh_classifier(dof_handler_ls, level_set);
  mesh_classifier.reclassify();
  set_fe_index(dof_handler_cut, mesh_classifier, false);
  dof_handler_cut.distribute_dofs(fe_collection);

  Quadrature<dim>           quadrature = QGauss<dim>(degree + 1);
  AffineConstraints<number> constraints;
  constraints.close();
  typename MatrixFree<dim, number, VectorizedArray<number>>::AdditionalData additional_data;
  additional_data.overlap_communication_computation = false;
  additional_data.hold_all_faces_to_owned_cells     = true;
  additional_data.mapping_update_flags =
    (update_values | update_JxW_values | update_quadrature_points);
  MatrixFree<dim, number, VectorizedArray<number>> matrix_free;
  matrix_free.reinit(mapping, dof_handler_cut, constraints, quadrature, additional_data);

  LinearAlgebra::distributed::Vector<number> solution;
  matrix_free.initialize_dof_vector(solution);
  VectorTools::interpolate(dof_handler_cut, StepFunction<dim>(), solution);

  MappingInfoVectorType<dim, number> mapping_info_cells(2);
  mapping_info_cells[0] =
    std::make_shared<NonMatching::MappingInfo<dim, dim, VectorizedArray<number>>>(
      mapping, update_values | update_JxW_values);
  mapping_info_cells[1] =
    std::make_shared<NonMatching::MappingInfo<dim, dim, VectorizedArray<number>>>(
      mapping, update_values | update_JxW_values);
  MappingInfoType<dim, number> dummy_mapping_info_surface(mapping,
                                                          dealii::update_values |
                                                            dealii::update_JxW_values);
  compute_intersected_quadrature(mapping_info_cells,
                                 dummy_mapping_info_surface,
                                 dof_handler_ls,
                                 level_set,
                                 matrix_free,
                                 degree,
                                 is_two_phase);

  const number norm = compute_cut_norm<dim>(
    solution, matrix_free, mapping_info_cells, is_two_phase, fe_q, 0, 0, NormType::L1_norm);
  if (Utilities::MPI::this_mpi_process(tria.get_communicator()) == 0)
    std::cout << "L1 norm: " << norm << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  test<2>();
}