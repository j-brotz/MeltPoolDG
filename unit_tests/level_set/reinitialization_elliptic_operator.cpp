#include <gtest/gtest.h>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/level_set/reinitialization_elliptic_operator.hpp>

#include <mpi.h>

#include <cmath>

namespace MeltPoolDG::LevelSet
{
  namespace
  {
    struct MPIInitializer
    {
      MPIInitializer()
      {
        int    argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
      }

      ~MPIInitializer()
      {
        MPI_Finalize();
      }
    };

    const MPIInitializer mpi_initializer;


    template <int dim>
    class QuadraticTestField : public dealii::Function<dim>
    {
    public:
      double
      value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
      {
        AssertIndexRange(component, 1);
        if constexpr (dim == 2)
          return p[0] * p[0] + p[1] * p[1] + 2.0 * p[0] + 3.0;
        else
          return p[0] * p[0] + 2.0 * p[0] + 3.0;
      }
    };

    template <int dim>
    struct EllipticOperatorFixture
    {
      using number          = double;
      using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;
      using MappingInfoType = CutUtil::MappingInfoType<dim, number>;

      dealii::Triangulation<dim>                                triangulation;
      dealii::FE_Q<dim>                                         fe{2};
      dealii::DoFHandler<dim>                                   dof_handler{triangulation};
      dealii::AffineConstraints<number>                         constraints;
      dealii::MappingQ1<dim>                                    mapping;
      MappingInfoType                                           mapping_info_surface;
      dealii::QGauss<dim>                                       quadrature{fe.degree + 1};
      MeltPoolDG::ScratchData<dim, dim, number>                 scratch_data;
      ReinitializationData<number>                              reinit_data;
      BlockVectorType                                           normal_vector;
      VectorType                                                level_set;
      std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;

      EllipticOperatorFixture()
        : mapping_info_surface(mapping,
                               dealii::update_values | dealii::update_gradients |
                                 dealii::update_JxW_values | dealii::update_normal_vectors)
        , scratch_data(MPI_COMM_WORLD, 0, true)
      {}

      void
      setup_rectangular_grid(const std::vector<unsigned int> &subdivisions,
                             const dealii::Point<dim>        &lower_left,
                             const dealii::Point<dim>        &upper_right)
      {
        dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          subdivisions,
                                                          lower_left,
                                                          upper_right);
        dof_handler.distribute_dofs(fe);

        constraints.close();

        scratch_data.reinit(
          mapping, {&dof_handler}, {&constraints}, {quadrature}, false, false, false);

        scratch_data.initialize_dof_vector(normal_vector, 0);

        dealii::Functions::ConstantFunction<dim> constant_level_set(1.0);
        dealii::VectorTools::interpolate(mapping, dof_handler, constant_level_set, level_set);
        mesh_classifier =
          std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(dof_handler, level_set);
      }

      ReinitializationEllipticOperator<dim, number>
      make_operator()
      {
        return ReinitializationEllipticOperator<dim, number>(
          scratch_data, reinit_data, 0, 0, mapping_info_surface, 0, mesh_classifier);
      }
    };

    template <int dim>
    void
    assemble_reference_laplace_matrix(const EllipticOperatorFixture<dim>     &fixture,
                                      dealii::TrilinosWrappers::SparseMatrix &reference_matrix)
    {
      dealii::TrilinosWrappers::SparsityPattern dsp(fixture.dof_handler.locally_owned_dofs(),
                                                    MPI_COMM_WORLD);
      dealii::DoFTools::make_sparsity_pattern(fixture.dof_handler, dsp, fixture.constraints, false);
      dsp.compress();

      reference_matrix.reinit(dsp);
      dealii::MatrixCreator::create_laplace_matrix(
        fixture.mapping,
        fixture.dof_handler,
        fixture.quadrature,
        reference_matrix,
        static_cast<const dealii::Function<dim, double> *>(nullptr),
        fixture.constraints);
    }

    template <int dim>
    void
    run_constructor_smoke_test()
    {
      EllipticOperatorFixture<dim> fixture;
      fixture.setup_rectangular_grid(std::vector<unsigned int>(dim, 2),
                                     dealii::Point<dim>(),
                                     dealii::Point<dim>(1.0));

      auto op = fixture.make_operator();
      ASSERT_NO_THROW(op.reinit());
    }

    template <int dim>
    void
    run_rhs_area_test()
    {
      using number     = double;
      using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

      EllipticOperatorFixture<dim> fixture;
      fixture.setup_rectangular_grid(std::vector<unsigned int>{2, 3},
                                     dealii::Point<dim>(),
                                     dealii::Point<dim>(2.0, 3.0));

      auto op = fixture.make_operator();
      ASSERT_NO_THROW(op.reinit());

      VectorType src;
      VectorType rhs;
      fixture.scratch_data.initialize_dof_vector(src, 0);
      fixture.scratch_data.initialize_dof_vector(rhs, 0);

      src = 0.0;
      op.create_rhs(rhs, src);

      const number expected_area = 6.0;
      EXPECT_NEAR(rhs.l1_norm(), expected_area, 1e-12);
    }

    template <int dim>
    void
    run_vmult_reference_test()
    {
      using number     = double;
      using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

      EllipticOperatorFixture<dim> fixture;
      fixture.setup_rectangular_grid(std::vector<unsigned int>{2, 3},
                                     dealii::Point<dim>(),
                                     dealii::Point<dim>(2.0, 3.0));

      auto op = fixture.make_operator();
      ASSERT_NO_THROW(op.reinit());

      VectorType src;
      VectorType dst_matrix_free;
      VectorType dst_reference;
      fixture.scratch_data.initialize_dof_vector(src, 0);
      fixture.scratch_data.initialize_dof_vector(dst_matrix_free, 0);
      fixture.scratch_data.initialize_dof_vector(dst_reference, 0);

      QuadraticTestField<dim> source_function;
      dealii::VectorTools::interpolate(fixture.mapping, fixture.dof_handler, source_function, src);
      src.update_ghost_values();

      op.vmult(dst_matrix_free, src);

      dealii::TrilinosWrappers::SparseMatrix reference_matrix;
      assemble_reference_laplace_matrix(fixture, reference_matrix);
      reference_matrix.vmult(dst_reference, src);

      VectorType diff = dst_matrix_free;
      diff -= dst_reference;

      const number reference_norm = dst_reference.l2_norm();
      EXPECT_NEAR(diff.l2_norm(), 0.0, std::max(number(1e-12), reference_norm * 1e-12));
    }
  } // namespace

  TEST(ReinitializationEllipticOperatorTest, ConstructorSmokeTest2D)
  {
    run_constructor_smoke_test<2>();
  }

  TEST(ReinitializationEllipticOperatorTest, RhsAreaTest2D)
  {
    run_rhs_area_test<2>();
  }

  TEST(ReinitializationEllipticOperatorTest, VmultMatchesReferenceMatrix2D)
  {
    run_vmult_reference_test<2>();
  }
} // namespace MeltPoolDG::LevelSet
