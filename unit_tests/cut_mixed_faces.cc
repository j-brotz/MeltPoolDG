// In this test, the function get_face_type() in 'cut/util.hpp' in combination with
// FEFaceEvaluation objects is tested, to correctly choose the active_fe_indices for mixed faces in
// cut applications. For a mixed face, one adjacent cell of the face is completeley inside the
// active phase and the other adjacent cell is cut.

// case 1:
//
//       liquid \  gas
//  |----|----|--\--|----|
// x=-1           \     x=1
//                phi

// case 2:
//
//       gas      /  liquid
//  |----|----|--/--|----|
// x=-1         /       x=1
//            phi


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>

#include <fstream>
#include <string>

using namespace dealii;
using namespace MeltPoolDG;

using Number              = double;
using VectorType          = LinearAlgebra::distributed::Vector<Number>;
using VectorizedArrayType = VectorizedArray<double>;

/******************************************************************************
 * Operator class
 *****************************************************************************/

template <int dim, int fe_degree>
class Operator
{
public:
  Operator();

  void
  reinit(const DoFHandler<dim> &dof_handler)
  {
    Quadrature<1> quadrature = QGauss<1>(fe_degree + 1);
    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData additional_data;
    additional_data.mapping_update_flags             = update_values;
    additional_data.mapping_update_flags_inner_faces = update_values;
    AffineConstraints<Number> constraints;

    data.reinit(mapping,
                std::vector<const DoFHandler<dim> *>{&dof_handler},
                std::vector<const AffineConstraints<Number> *>{&constraints},
                std::vector<Quadrature<1>>{{quadrature}},
                additional_data);
  };

  void
  initialize_vector(VectorType &vec, const DoFHandler<dim> &dof_handler)
  {
    data.initialize_dof_vector(vec);
  };

  void
  vmult(VectorType &vec) const
  {
    typedef std::function<void(const MatrixFree<dim, Number> &,
                               LinearAlgebra::distributed::Vector<Number>       &dst,
                               const LinearAlgebra::distributed::Vector<Number> &src,
                               const std::pair<unsigned int, unsigned int> &)>
      local_applier_type;

    local_applier_type cell          = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_cell);
    local_applier_type face          = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_face);
    local_applier_type boundary_face = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_boundary_face);
    data.loop(cell,
              face,
              boundary_face,
              vec,
              vec,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
              MatrixFree<dim, Number>::DataAccessOnFaces::gradients);
  }

  void
  local_apply_cell(const dealii::MatrixFree<dim, Number> &,
                   VectorType                          &dst,
                   const VectorType                    &src,
                   const std::pair<unsigned, unsigned> &cell_range) const
  {}

  void
  local_apply_face(const dealii::MatrixFree<dim, Number> &,
                   VectorType                          &dst,
                   const VectorType                    &src,
                   const std::pair<unsigned, unsigned> &face_range) const
  {
    const auto              face_category = data.get_face_range_category(face_range);
    const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);

    switch (face_type)
      {
        case CutUtil::FaceType::mixed_face_liquid_intersected:
          for (unsigned int face = face_range.first; face < face_range.second; ++face)
            ++counter_mixed_face_liquid_intersected;
          break;

        case CutUtil::FaceType::mixed_face_intersected_liquid:
          for (unsigned int face = face_range.first; face < face_range.second; ++face)
            ++counter_mixed_face_intersected_liquid;
          break;

        case CutUtil::FaceType::mixed_face_gas_intersected:
          for (unsigned int face = face_range.first; face < face_range.second; ++face)
            ++counter_mixed_face_gas_intersected;
          break;

        case CutUtil::FaceType::mixed_face_intersected_gas:
          for (unsigned int face = face_range.first; face < face_range.second; ++face)
            ++counter_mixed_face_intersected_gas;
          break;

        default:
          break;
      }
  }

  void
  local_apply_boundary_face(const dealii::MatrixFree<dim, Number> &,
                            VectorType                          &dst,
                            const VectorType                    &src,
                            const std::pair<unsigned, unsigned> &boundary_face_range) const
  {}

  // counters
  mutable unsigned int counter_mixed_face_liquid_intersected = 0;
  mutable unsigned int counter_mixed_face_intersected_liquid = 0;
  mutable unsigned int counter_mixed_face_gas_intersected    = 0;
  mutable unsigned int counter_mixed_face_intersected_gas    = 0;

private:
  MatrixFree<dim, Number, VectorizedArrayType> data;
  MappingQ<dim>                                mapping;
};


template <int dim, int fe_degree>
Operator<dim, fe_degree>::Operator()
  : mapping(fe_degree)
{}


/******************************************************************************
 * test function
 *****************************************************************************/
template <int dim, bool is_dg, int fe_degree, bool liquid_phase_is_left>
void
test()
{
  ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
  pcout << std::endl;
  pcout << "Test case with liquid_phase_is_left: " << liquid_phase_is_left << std::endl;

  // create mesh with 4 elements
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -1.0, 1.0);
  tria.refine_global(2);

  // create level-set
  DoFHandler<dim> ls_dof_handler(tria);
  ls_dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));
  VectorType ls_vector;
  ls_vector.reinit(ls_dof_handler.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(ls_dof_handler),
                   tria.get_communicator());
  Point<dim> p;
  // choose interface in the center of the third element
  p[0] = 0.25;
  dealii::Tensor<1, dim> normal;
  normal[0] = 1.;
  const Functions::SignedDistance::Plane<dim> signed_distance(p, normal);
  VectorTools::interpolate(ls_dof_handler, signed_distance, ls_vector);
  if (liquid_phase_is_left)
    ls_vector *= -1.;
  ls_vector.update_ghost_values();

  // create mesh classifier
  NonMatching::MeshClassifier<dim> mesh_classifier(ls_dof_handler, ls_vector);
  mesh_classifier.reclassify();

  // create two-phase solution vector
  DoFHandler<dim> dof_handler(tria);
  CutUtil::set_fe_index<dim>(dof_handler, mesh_classifier, false);

  FE_Nothing<dim>                                                fe_n;
  hp::FECollection<dim>                                          fe_collection;
  typename std::conditional<is_dg, FE_DGQ<dim>, FE_Q<dim>>::type fe_q(fe_degree);

  fe_collection.push_back(FESystem<dim>(fe_q, 1, fe_n, 1)); // liquid
  fe_collection.push_back(FESystem<dim>(fe_q, 1, fe_q, 1)); // intersected
  fe_collection.push_back(FESystem<dim>(fe_n, 1, fe_q, 1)); // gas
  dof_handler.distribute_dofs(fe_collection);

  // setup operator class
  Operator<dim, fe_degree> test_operator;

  // initialize matrix-free
  test_operator.reinit(dof_handler);

  // initialize solution vector
  VectorType solution;
  test_operator.initialize_vector(solution, dof_handler);
  solution = 1.;

  // Check mixed face categories
  test_operator.vmult(solution);

  // Output how often the different types of mixed faces are visited
  std::cout << "Number of mixed_face_liquid_intersected: "
            << test_operator.counter_mixed_face_liquid_intersected << std::endl;
  std::cout << "Number of mixed_face_intersected_liquid: "
            << test_operator.counter_mixed_face_intersected_liquid << std::endl;
  std::cout << "Number of mixed_face_gas_intersected: "
            << test_operator.counter_mixed_face_gas_intersected << std::endl;
  std::cout << "Number of mixed_face_intersected_gas: "
            << test_operator.counter_mixed_face_intersected_gas << std::endl;

  // output solution and level-set
#if false
  DataOut<dim> data_out;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  data_out.add_data_vector(ls_dof_handler, ls_vector, "level_set");
  data_out.add_data_vector(dof_handler, solution, "solution");

  data_out.build_patches();
  std::ofstream out("cut_mixed_faces.vtu");
  data_out.write_gnuplot(out);
#endif
}

int
main(int argc, char *argv[])
{
  // case 1: liquid phase is left:
  test<1 /*dim*/, true /*is_dg*/, 1 /*fe_degreee*/, true /*liquid_phase_is_left*/>();

  // case 2: liquid phase is right:
  test<1 /*dim*/, true /*is_dg*/, 1 /*fe_degreee*/, false /*liquid_phase_is_left*/>();

  return 0;
}
