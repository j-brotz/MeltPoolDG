// In this test, 'cut_solution_transfer.hpp' and 'cut_util.hpp' are
// tested.

// Following options are tested:
// - single-phase and two-phase
// - cutFEM and cutDG
// - fe-degree 1 and 2
// - scalar-valued and vector-valued solutions
// - two and three space dimensions
// - default vector initialization and initialization with matrix-free object

// In 'cut_solution_transfer.hpp', the solution projection between function spaces
// according to the PhD thesis of Schott (2017) 3.6.3.2 is computed.

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

#include <meltpooldg/cut/cut_solution_transfer.hpp>
#include <meltpooldg/cut/cut_util.hpp>

#include <mpi.h>

#include <fstream>
#include <string>

using namespace dealii;
using namespace MeltPoolDG;

using Number              = double;
using VectorType          = LinearAlgebra::distributed::Vector<Number>;
using VectorizedArrayType = VectorizedArray<double>;


/******************************************************************************
 * Initial conditions
 *****************************************************************************/

template <int dim, int n_solution_components>
class InitialConditionFunction : public Function<dim>
{
public:
  InitialConditionFunction(const unsigned int n_total_components)
    : Function<dim>(n_total_components)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    return value<double>(p, component);
  }

  template <typename Number>
  Number
  value(const Point<dim, Number> &p, const unsigned int component) const
  {
    Number return_value = 1.;

    if (n_solution_components > 1)
      {
        if (component == 0 || component == 3)
          for (unsigned int i = 0; i < dim; ++i)
            return_value *= 2. * std::cos(p[i] * numbers::PI / 2.);
        else if (component == 1 || component == 4)
          for (unsigned int i = 0; i < dim; ++i)
            return_value *= std::cos(p[i] * numbers::PI / 2.);
        else if (component == 2 || component == 5)
          for (unsigned int i = 0; i < dim; ++i)
            return_value *= 0.5 * std::cos(p[i] * numbers::PI / 2.);
      }
    else
      {
        for (unsigned int i = 0; i < dim; ++i)
          return_value *= std::cos(p[i] * numbers::PI / 2.);
      }

    return return_value;
  }
};



/******************************************************************************
 * Reinit class
 *****************************************************************************/

template <int dim, int fe_degree, bool is_matrix_free>
class ReinitOperator
{
public:
  ReinitOperator();

  void
  reinit(const DoFHandler<dim> &dof_handler);

  /**
   * Lambda function for matrix-free reinitialization.
   */
  using reinit_matrix_free = std::function<void(const DoFHandler<dim> &)>;

  reinit_matrix_free
  get_reinit_matrix_free()
  {
    return [&](const DoFHandler<dim> &dof_handler) {
      Quadrature<1> quadrature = QGauss<1>(fe_degree + 1);
      typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData additional_data;
      additional_data.mapping_update_flags =
        update_gradients | update_values | update_quadrature_points;
      if (fe_degree == 1)
        additional_data.mapping_update_flags_inner_faces = update_values | update_gradients;
      else if (fe_degree == 2)
        additional_data.mapping_update_flags_inner_faces =
          update_values | update_gradients | update_hessians;
      else
        Assert(false, ExcNotImplemented());

      additional_data.hold_all_faces_to_owned_cells = true;

      AffineConstraints<Number> constraints;

      data.reinit(mapping,
                  std::vector<const DoFHandler<dim> *>{&dof_handler},
                  std::vector<const AffineConstraints<Number> *>{&constraints},
                  std::vector<Quadrature<1>>{{quadrature}},
                  additional_data);
    };
  }

  void
  initialize_vector(VectorType &, const DoFHandler<dim> &dof_handler);

  /**
   * Lambda function for vector reinitialization.
   */
  using reinit_vector = std::function<void(VectorType &, const DoFHandler<dim> &)>;

  reinit_vector
  get_reinit_vector()
  {
    return [&](VectorType &vec, const DoFHandler<dim> &dof_handler) {
      if (is_matrix_free)
        data.initialize_dof_vector(vec);
      else
        vec.reinit(dof_handler.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(dof_handler),
                   dof_handler.get_communicator());
    };
  }

private:
  MatrixFree<dim, Number, VectorizedArrayType> data;
  MappingQ<dim>                                mapping;
};


template <int dim, int fe_degree, bool is_matrix_free>
ReinitOperator<dim, fe_degree, is_matrix_free>::ReinitOperator()
  : mapping(fe_degree)
{}


template <int dim, int fe_degree, bool is_matrix_free>
void
ReinitOperator<dim, fe_degree, is_matrix_free>::reinit(const DoFHandler<dim> &dof_handler)
{
  reinit_matrix_free lambda_reinit_matrix_free = get_reinit_matrix_free();
  lambda_reinit_matrix_free(dof_handler);
}


template <int dim, int fe_degree, bool is_matrix_free>
void
ReinitOperator<dim, fe_degree, is_matrix_free>::initialize_vector(
  VectorType            &vec,
  const DoFHandler<dim> &dof_handler)
{
  reinit_vector lambda_reinit_vector = get_reinit_vector();
  lambda_reinit_vector(vec, dof_handler);
}


/******************************************************************************
 * test
 *****************************************************************************/
template <int  dim,
          bool is_dg,
          bool is_two_phase,
          int  n_solution_components,
          bool is_matrix_free,
          int  fe_degree>
void
test()
{
  ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
  pcout << std::endl;
  pcout << "  start test with DG=" << is_dg << ", is_two_phase=" << is_two_phase
        << ", components of solution=" << n_solution_components << ", dimension=" << dim
        << ", is_matrix_free=" << is_matrix_free << ", fe_degree=" << fe_degree << std::endl;

  // parameters
  const double gamma_degree_0 = 0.5; // ghost-penalty parameter for jump in the 0. normal derivative
  const double gamma_degree_1 = 0.5; // ghost-penalty parameter for jump in the 1. normal derivative
  const double gamma_degree_2 = 0.5; // ghost-penalty parameter for jump in the 2. normal derivative

  // create mesh
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1.0, 1.0);
  tria.refine_global(4 - fe_degree);

  // create level-set DoFHandler and allocate vector
  DoFHandler<dim> ls_dof_handler(tria);
  ls_dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));
  VectorType ls_vector;
  ls_vector.reinit(ls_dof_handler.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(ls_dof_handler),
                   tria.get_communicator());

  // create mesh classifier
  NonMatching::MeshClassifier<dim> mesh_classifier(ls_dof_handler, ls_vector);
  NonMatching::MeshClassifier<dim> mesh_classifier_old(ls_dof_handler, ls_vector);

  // create system
  FE_Nothing<dim>                                                fe_n;
  hp::FECollection<dim>                                          fe_collection;
  typename std::conditional<is_dg, FE_DGQ<dim>, FE_Q<dim>>::type fe_q(fe_degree);

  if (n_solution_components > 1)
    {
      if (!is_two_phase)
        {
          FESystem<dim> fe_inside_intersected(fe_q, n_solution_components);
          FESystem<dim> fe_outside(fe_n, n_solution_components);

          fe_collection.push_back(fe_inside_intersected); // inside
          fe_collection.push_back(fe_inside_intersected); // intersected
          fe_collection.push_back(fe_outside);            // outside
        }
      else
        {
          FESystem<dim> fe_inside(fe_q, n_solution_components, fe_n, n_solution_components);
          FESystem<dim> fe_intersected(fe_q, n_solution_components, fe_q, n_solution_components);
          FESystem<dim> fe_outside(fe_n, n_solution_components, fe_q, n_solution_components);

          fe_collection.push_back(fe_inside);      // inside
          fe_collection.push_back(fe_intersected); // intersected
          fe_collection.push_back(fe_outside);     // outside
        }
    }
  else
    {
      if (!is_two_phase)
        {
          fe_collection.push_back(fe_q); // inside
          fe_collection.push_back(fe_q); // intersected
          fe_collection.push_back(fe_n); // outside
        }
      else
        {
          fe_collection.push_back(FESystem<dim>(fe_q, 1, fe_n, 1)); // inside
          fe_collection.push_back(FESystem<dim>(fe_q, 1, fe_q, 1)); // intersected
          fe_collection.push_back(FESystem<dim>(fe_n, 1, fe_q, 1)); // outside
        }
    }

  ///////////////////////////////////////////////////////////////////
  // Prepare initial solution
  ///////////////////////////////////////////////////////////////////
  DoFHandler<dim> dof_handler(tria);

  Point<dim>                                   p;
  const Number                                 radius = 0.51;
  const Functions::SignedDistance::Sphere<dim> signed_distance(p, radius);
  VectorTools::interpolate(ls_dof_handler, signed_distance, ls_vector);
  ls_vector *= -1.;

  ls_vector.update_ghost_values();

  mesh_classifier_old.reclassify();

  CutUtil::set_fe_index<dim>(dof_handler, mesh_classifier_old, false);

  dof_handler.distribute_dofs(fe_collection);

  // setup reinit class
  ReinitOperator<dim, fe_degree, is_matrix_free> reinit_operator;

  // initialize reinit class
  reinit_operator.reinit(dof_handler);

  // initialize solution vector
  VectorType solution;
  reinit_operator.initialize_vector(solution, dof_handler);
  solution = 0.;

  // interpolate initial solution from function

  // total number of solution components
  unsigned int n_total_components = n_solution_components;
  if (is_two_phase)
    n_total_components = 2 * n_solution_components;

  InitialConditionFunction<dim, n_solution_components> initial_condition_function(
    n_total_components);
  VectorTools::interpolate(dof_handler, initial_condition_function, solution);

  solution.update_ghost_values();
  pcout << "  solution: " << solution.l2_norm() << std::endl;

  // output solution with old interface/boundary position
#if false
  DataOut<dim> data_out;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  data_out.add_data_vector(ls_dof_handler, ls_vector, "level_set");
  data_out.add_data_vector(dof_handler, solution, "solution");

  Vector<Number> mpi_owner(tria.n_active_cells());
  mpi_owner = Utilities::MPI::this_mpi_process(tria.get_communicator());
  data_out.add_data_vector(mpi_owner, "owner");

  data_out.build_patches();
  const std::string filename = "moving_grid.vtu";
  data_out.write_vtu_in_parallel(filename, tria.get_communicator());
#endif

  ///////////////////////////////////////////////////////////////////
  // Move interface
  ///////////////////////////////////////////////////////////////////
  {
    ls_vector.zero_out_ghost_values();

    Point<dim>   p;
    const Number radius = 0.51 + GridTools::minimal_cell_diameter(tria) * 0.5;
    const Functions::SignedDistance::Sphere<dim> signed_distance(p, radius);
    VectorTools::interpolate(ls_dof_handler, signed_distance, ls_vector);
    ls_vector *= -1.;

    ls_vector.update_ghost_values();
    mesh_classifier.reclassify();
  }

  ///////////////////////////////////////////////////////////////////////
  // transfer solution according to the new interface position
  ///////////////////////////////////////////////////////////////////////

  // verbosity level for solution transfer output (0: nothing, 1: some output, 2: more details for
  // testing)
  unsigned int verbosity = 0;

  // consider dummy-data input for 'gamma_degree_0' for the cg-case and
  // for 'gamma_degree_2' for polynomial degree 1.
  CutUtil::SolutionTransferOperator<dim, Number> solution_transfer_operator(
    gamma_degree_0, gamma_degree_1, gamma_degree_2, is_two_phase, verbosity);

  solution.update_ghost_values();

  if (is_matrix_free)
    solution_transfer_operator.reinit(dof_handler,
                                      tria,
                                      solution,
                                      mesh_classifier_old,
                                      mesh_classifier,
                                      reinit_operator.get_reinit_vector(),
                                      reinit_operator.get_reinit_matrix_free());
  else
    solution_transfer_operator.reinit(dof_handler,
                                      tria,
                                      solution,
                                      mesh_classifier_old,
                                      mesh_classifier,
                                      reinit_operator.get_reinit_vector());

  // reinit transferred solution vector
  VectorType solution_gp_extrapolated;
  reinit_operator.initialize_vector(solution_gp_extrapolated, dof_handler);
  solution_gp_extrapolated = 0.;

  solution_gp_extrapolated.swap(solution_transfer_operator.get_updated_solution());

  pcout << "  solution after gp-extrapolation, l2-norm: " << solution_gp_extrapolated.l2_norm()
        << std::endl;

  // output gp-extrapolated solution for new interface/boundary position
#if false
  {
    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.add_data_vector(dof_handler, solution_gp_extrapolated, "solution_gp_extrapolated");
    data_out.add_data_vector(ls_dof_handler, ls_vector, "level_set");

    Vector<Number> mpi_owner(tria.n_active_cells());
    mpi_owner = Utilities::MPI::this_mpi_process(tria.get_communicator());
    data_out.add_data_vector(mpi_owner, "owner");

    data_out.build_patches();
    const std::string filename = "moving_grid_new_gp_extrapolated.vtu";
    data_out.write_vtu_in_parallel(filename, tria.get_communicator());
  }
#endif
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  /////////////////////////////////////////////
  ////////// tests for fe_degree = 1 //////////
  /////////////////////////////////////////////

  /////////////////////////////////////////////
  ///// tests for scalar-valued solutions /////
  /////////////////////////////////////////////

  // use default vector initialization

  // 2D
  // two-phase
  test<2, false, true, 1, false, 1>();
  test<2, true, true, 1, false, 1>();
  // one-phase
  test<2, false, false, 1, false, 1>();
  test<2, true, false, 1, false, 1>();

  // 3D
  // two-phase
  test<3, false, true, 1, false, 1>();
  test<3, true, true, 1, false, 1>();
  // one-phase
  test<3, false, false, 1, false, 1>();
  test<3, true, false, 1, false, 1>();

  // use vector initialization with matrix-free object

  // 2D
  // two-phase
  test<2, false, true, 1, true, 1>();
  test<2, true, true, 1, true, 1>();
  // one-phase
  test<2, false, false, 1, true, 1>();
  test<2, true, false, 1, true, 1>();

  // 3D
  // two-phase
  test<3, false, true, 1, true, 1>();
  test<3, true, true, 1, true, 1>();
  // one-phase
  test<3, false, false, 1, true, 1>();
  test<3, true, false, 1, true, 1>();

  /////////////////////////////////////////////
  ///// tests for vector-valued solutions /////
  /////////////////////////////////////////////

  // use default vector initialization

  // 2D
  // two-phase
  test<2, false, true, 3, false, 1>();
  test<2, true, true, 3, false, 1>();
  // one-phase
  test<2, false, false, 3, false, 1>();
  test<2, true, false, 3, false, 1>();

  // 3D
  // two-phase
  test<3, false, true, 3, false, 1>();
  test<3, true, true, 3, false, 1>();
  // one-phase
  test<3, false, false, 3, false, 1>();
  test<3, true, false, 3, false, 1>();

  // use vector initialization with matrix-free object

  // 2D
  // two-phase
  test<2, false, true, 3, true, 1>();
  test<2, true, true, 3, true, 1>();
  // one-phase
  test<2, false, false, 3, true, 1>();
  test<2, true, false, 3, true, 1>();

  // 3D
  // two-phase
  test<3, false, true, 3, true, 1>();
  test<3, true, true, 3, true, 1>();
  // one-phase
  test<3, false, false, 3, true, 1>();
  test<3, true, false, 3, true, 1>();

  /////////////////////////////////////////////
  ////////// tests for fe_degree = 2 //////////
  /////////////////////////////////////////////

  /////////////////////////////////////////////
  ///// tests for scalar-valued solutions /////
  /////////////////////////////////////////////

  // use vector initialization with matrix-free object

  // 2D
  // two-phase
  test<2, false, true, 1, true, 2>();
  test<2, true, true, 1, true, 2>();
  // one-phase
  test<2, false, false, 1, true, 2>();
  test<2, true, false, 1, true, 2>();

  /////////////////////////////////////////////
  ///// tests for vector-valued solutions /////
  /////////////////////////////////////////////

  // use vector initialization with matrix-free object

  // 2D
  // two-phase
  test<2, false, true, 3, true, 2>();
  test<2, true, true, 3, true, 2>();
  // one-phase
  test<2, false, false, 3, true, 2>();
  test<2, true, false, 3, true, 2>();

  return 0;
}
