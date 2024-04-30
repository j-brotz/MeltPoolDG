#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <iostream>

using namespace dealii;
using namespace MeltPoolDG;

using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int dim>
class InitializePhi1 : public Function<dim>
{
public:
  InitializePhi1(const Point<dim> &center, const double &radius)
    : Function<dim>()
    , sphere(center, radius)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;

    return UtilityFunctions::CharacteristicFunctions::sgn(-sphere.value(p));
  }

  const Functions::SignedDistance::Sphere<dim> sphere;
};

template <int dim>
class InitializePhi2 : public Function<dim>
{
public:
  InitializePhi2(const Point<dim> &center, const double &radius)
    : Function<dim>()
    , sphere(center, radius)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;

    return UtilityFunctions::CharacteristicFunctions::sgn(sphere.value(p));
  }

  Point<dim>                                   center;
  double                                       radius;
  const Functions::SignedDistance::Sphere<dim> sphere;
};

template <int dim>
class InitializePhi3 : public Function<dim>
{
public:
  InitializePhi3(const Point<dim> &center, const double &radius)
    : Function<dim>()
    , sphere(center, radius)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;

    return UtilityFunctions::CharacteristicFunctions::heaviside(
      UtilityFunctions::CharacteristicFunctions::sgn(-sphere.value(p)), 0);
  }

  const Functions::SignedDistance::Sphere<dim> sphere;
};

template <int dim>
class InitializePhi4 : public Function<dim>
{
public:
  InitializePhi4(const Point<dim> &center, const double &radius)
    : Function<dim>()
    , sphere(center, radius)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;

    return UtilityFunctions::CharacteristicFunctions::heaviside(
      -UtilityFunctions::CharacteristicFunctions::sgn(-sphere.value(p)), 0);
  }

  const Functions::SignedDistance::Sphere<dim> sphere;
};

template <int dim>
class InitializePhi5 : public Function<dim>
{
public:
  InitializePhi5(const Point<dim> &center, const double &radius)
    : Function<dim>()
    , sphere(center, radius)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;

    return 5 * UtilityFunctions::CharacteristicFunctions::sgn(-sphere.value(p)) + 1.0;
  }

  Point<dim>                                   center;
  double                                       radius;
  const Functions::SignedDistance::Sphere<dim> sphere;
};


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_comm(MPI_COMM_WORLD);

  const int dim    = 2;
  const int degree = 1;

  Point<2>     center1(0.5, 0.5);
  Point<2>     center2(0.3, 0.3);
  const double radius = 0.25;

  // triangulation
  auto triangulation = parallel::distributed::Triangulation<dim>(mpi_comm);

  GridGenerator::hyper_cube(triangulation);

  triangulation.refine_global(5);

  auto q_gauss = QGauss<dim>(degree + 1);
  auto fe_q    = FE_Q<dim>(degree);
  auto mapping = MappingFE<dim>(fe_q);

  DoFHandler<dim> dof_handler;
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;

  typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values | dealii::update_quadrature_points;

  MatrixFree<dim, double, VectorizedArray<double>> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, q_gauss, additional_data);

  const auto run_test = [&](const auto  &level_set_1,
                            const auto  &level_set_2,
                            const double level_set_interior_value = 1.0,
                            const double level_set_exterior_value = -1.0,
                            std::string  pv_name                  = "test") -> auto {
    level_set_1.update_ghost_values();
    level_set_2.update_ghost_values();

    auto merged_level_set_union =
      LevelSet::Tools::merge_two_indicator_fields(level_set_1,
                                                  level_set_2,
                                                  BooleanType::Union,
                                                  level_set_interior_value,
                                                  level_set_exterior_value);

    auto merged_level_set_intersect =
      LevelSet::Tools::merge_two_indicator_fields(level_set_1,
                                                  level_set_2,
                                                  BooleanType::Intersection,
                                                  level_set_interior_value,
                                                  level_set_exterior_value);

    auto merged_level_set_subtract =
      LevelSet::Tools::merge_two_indicator_fields(level_set_1,
                                                  level_set_2,
                                                  BooleanType::Subtraction,
                                                  level_set_interior_value,
                                                  level_set_exterior_value);

    merged_level_set_union.update_ghost_values();
    merged_level_set_intersect.update_ghost_values();
    merged_level_set_subtract.update_ghost_values();

    dealii::ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);
    pcout << "level_set_1: " << level_set_1.l2_norm() << std::endl;
    pcout << "level_set_2: " << level_set_2.l2_norm() << std::endl;
    pcout << "level_set_1 ∪ level_set_2 : " << merged_level_set_union.l2_norm() << std::endl;
    pcout << "level_set_1 ∩ level_set_2 : " << merged_level_set_intersect.l2_norm() << std::endl;
    pcout << "level_set_1 - level_set_2 : " << merged_level_set_subtract.l2_norm() << std::endl;

#if 0
    // write paraview output
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.add_data_vector(level_set_1, "level_set_1");
    data_out.add_data_vector(level_set_2, "level_set_2");
    data_out.add_data_vector(merged_level_set_union, "merged_level_set_union");
    data_out.add_data_vector(merged_level_set_intersect, "merged_level_set_intersect");
    data_out.add_data_vector(merged_level_set_subtract, "merged_level_set_subtract");

    data_out.build_patches(mapping);
    data_out.write_vtu_with_pvtu_record("./", pv_name, 0, mpi_comm);
#else
    (void)pv_name;
#endif
    level_set_1.zero_out_ghost_values();
    level_set_2.zero_out_ghost_values();
  };


  VectorType level_set_1, level_set_2;
  matrix_free.initialize_dof_vector(level_set_1, 0);
  matrix_free.initialize_dof_vector(level_set_2, 0);

  /*
   * Test with level set interior = 1.0 and exterior = -1.0
   */
  VectorTools::interpolate(mapping, dof_handler, InitializePhi1<dim>(center1, radius), level_set_1);
  VectorTools::interpolate(mapping, dof_handler, InitializePhi1<dim>(center2, radius), level_set_2);
  run_test(level_set_1, level_set_2, 1.0, -1.0, "test_1");

  /*
   * Test with level set interior = -1.0 and exterior = 1.0
   */
  VectorTools::interpolate(mapping, dof_handler, InitializePhi2<dim>(center1, radius), level_set_1);
  VectorTools::interpolate(mapping, dof_handler, InitializePhi2<dim>(center2, radius), level_set_2);
  run_test(level_set_1, level_set_2, -1.0, 1.0, "test_2");

  /*
   * Test with level set interior = 1.0 and exterior = 0.0
   */
  VectorTools::interpolate(mapping, dof_handler, InitializePhi3<dim>(center1, radius), level_set_1);
  VectorTools::interpolate(mapping, dof_handler, InitializePhi3<dim>(center2, radius), level_set_2);
  run_test(level_set_1, level_set_2, 1.0, 0.0, "test_3");

  /*
   * Test with level set interior = 0.0 and exterior = 1.0
   */
  VectorTools::interpolate(mapping, dof_handler, InitializePhi4<dim>(center1, radius), level_set_1);
  VectorTools::interpolate(mapping, dof_handler, InitializePhi4<dim>(center2, radius), level_set_2);
  run_test(level_set_1, level_set_2, 0.0, 1.0, "test_4");

  /*
   * Test with level set interior = 6.0 and exterior = -4.0
   */
  VectorTools::interpolate(mapping, dof_handler, InitializePhi5<dim>(center1, radius), level_set_1);
  VectorTools::interpolate(mapping, dof_handler, InitializePhi5<dim>(center2, radius), level_set_2);
  run_test(level_set_1, level_set_2, 6.0, -4.0, "test_5");

  return 0;
}
