#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>

#include "meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp"
#include "meltpooldg/time_integration/time_stepping_data.hpp"
#include "meltpooldg/utilities/matrix_free_util.hpp"
#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/matrix_free_particle_cache.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/fe_util.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include <memory>
#include <vector>

constexpr int dim = 3;

class CompressibleFlowOperatorFixture : public benchmark::Fixture
{
public:
  /**
   * @brief Holds all operator-specific objects and functions.
   *
   * This struct is used to simplify data management. By grouping all related resources under
   * RAII, they are automatically released when the struct is destroyed. This ensures safe and
   * error-free cleanup in the TearDown function, avoiding potential memory issues from forgotten
   * deallocations.
   */
  struct Data
  {
    /**
     * @brief Constructor that sets up all necessary components required for finally setting up
     * the flow operator. Initializes the scratch data, including triangulation, DoF handler, and
     * other required components. Also sets the initial and boundary conditions.
     *
     * @param state Google Benchmark state object used to determine the number of cells in each
     * spatial direction of the cubic domain.
     */
    explicit Data(const benchmark::State &state)
      : triangulation(MPI_COMM_WORLD,
                      dealii::Triangulation<dim>::MeshSmoothing::none,
                      dealii::parallel::distributed::Triangulation<
                        dim>::Settings::construct_multigrid_hierarchy)
      , scratch_data(MPI_COMM_WORLD, 0, true)
      , timer(std::cout,
              dealii::TimerOutput::OutputFrequency::never,
              dealii::TimerOutput::wall_times)
    {
      set_simulation_parameters();
      setup_rectangular_triangulation(std::vector<unsigned int>{2, 2, 4},
                                      dealii::Point<dim>(0., 0., 0.),
                                      dealii::Point<dim>(2e-4, 2e-4, 4e-4),
                                      4);

      dof_handler.reinit(triangulation);
      MeltPoolDG::FiniteElementUtils::distribute_dofs<dim, dim + 2>(fe_data, dof_handler);

      setup_scratch_data();



      set_initial_conditions();
    }

    dealii::parallel::distributed::Triangulation<dim>  triangulation;
    dealii::DoFHandler<dim>                            dof_handler;
    MeltPoolDG::FiniteElementData                      fe_data;
    dealii::AffineConstraints<double>                  affine_constraints;
    MeltPoolDG::ScratchData<dim, dim, double>          scratch_data;
    dealii::LinearAlgebra::distributed::Vector<double> solution;
    MeltPoolDG::BrinkmanPenalizationData<double>       brinkman_data;
    std::unique_ptr<
      MeltPoolDG::BrinkmanObstacleForce<dim, double, MeltPoolDG::SphericalParticle<dim, double>>>
                                     brinkman_obstacle_force;
    MeltPoolDG::ObstacleData<double> obstacle_data;
    std::unique_ptr<
      MeltPoolDG::ObstacleField<dim, double, MeltPoolDG::SphericalParticle<dim, double>>>
                        obstacle_field;
    dealii::TimerOutput timer;

  private:
    void
    setup_rectangular_triangulation(const std::vector<unsigned int> &n_base_cells,
                                    const dealii::Point<dim>        &bottom_left,
                                    const dealii::Point<dim>        &top_right,
                                    unsigned int                     n_global_refinements)
    {
      dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        n_base_cells,
                                                        bottom_left,
                                                        top_right);

      triangulation.refine_global(n_global_refinements);
    }

    void
    setup_particle_field()
    {
      obstacle_data.obstacle_state_input_file =
        std::string(MPDG_BENCHMARK_DATA_DIR) + "/input_file_28_particles.csv";
      obstacle_field = std::make_unique<
        MeltPoolDG::ObstacleField<dim, double, MeltPoolDG::SphericalParticle<dim, double>>>(
        obstacle_data, triangulation, dealii::MappingQ1<dim>(), timer);
    }

    /**
     * @brief Initialize scratch data and flow-specific scratch data.
     *
     * Sets up the main scratch data object (including mapping, quadrature,
     * DoF handler, and constraints) and constructs the flow scratch data.
     */
    void
    setup_scratch_data()
    {
      scratch_data.set_mapping(MeltPoolDG::FiniteElementUtils::create_mapping<dim>(fe_data));

      unsigned int quad_index = scratch_data.attach_quadrature(
        MeltPoolDG::FiniteElementUtils::create_quadrature<dim>(fe_data));

      unsigned int dof_index = scratch_data.attach_dof_handler(dof_handler);

      scratch_data.attach_constraint_matrix(affine_constraints);
      scratch_data.create_partitioning();
      scratch_data.build(true, true, false, false);

      obstacle_data.obstacle_state_input_file =
        std::string(MPDG_BENCHMARK_DATA_DIR) + "/input_file_28_particles.csv";
      obstacle_field = std::make_unique<
        MeltPoolDG::ObstacleField<dim, double, MeltPoolDG::SphericalParticle<dim, double>>>(
        obstacle_data,
        triangulation,
        dealii::MappingQ1<dim>(),
        timer);

      auto matrix_free_cell_batch_particle_cache =
        std::make_shared<MeltPoolDG::Particles::MatrixFreeCellBatchParticleCache<dim, double>>(
          MeltPoolDG::MatrixFreeContext<dim, double>(scratch_data.get_matrix_free(), 0, 0));

      brinkman_obstacle_force = std::make_unique<
        MeltPoolDG::BrinkmanObstacleForce<dim, double, MeltPoolDG::SphericalParticle<dim, double>>>(
        *obstacle_field,
        solution,
        MeltPoolDG::MatrixFreeContext<dim, double>(scratch_data.get_matrix_free(),
                                                   dof_index,
                                                   quad_index),
        brinkman_data,
        matrix_free_cell_batch_particle_cache);
    }

    /**
     * @brief Set simulation parameters explicitly, as defaults are not suitable.
     */
    void
    set_simulation_parameters()
    {
      fe_data.type   = MeltPoolDG::FiniteElementType::FE_DGQ;
      fe_data.degree = 1;
    }

    /**
     * @brief Set initial and boundary conditions. Applies inflow and outflow boundary conditions
     * based on the predefined inflow field. All other boundaries use no-slip conditions.
     */
    void
    set_initial_conditions()
    {
      scratch_data.get_matrix_free().initialize_dof_vector(solution);
      solution.add(std::abs(std::rand()));
    }
  };

  // Prevent compiler warnings about hiding overloaded virtual functions (-Woverloaded-virtual).
  using benchmark::Fixture::SetUp;
  using benchmark::Fixture::TearDown;


  /**
   * @brief Fixture setup function called before each benchmark run. Initializes the
   * benchmark-specific data structure using the benchmark state.
   *
   * @param state Reference to the current benchmark state.
   */
  void
  SetUp(benchmark::State &state) override
  {
    data = std::make_unique<Data>(state);
  }


  /**
   * @brief Fixture teardown function called after each benchmark run. Cleans up and deallocates
   * the benchmark-specific data, i.e., the data object struct.
   */
  void
  TearDown(benchmark::State &) override
  {
    data.reset();
  }

  // Holds all data for the compressible flow operator.
  std::unique_ptr<Data> data;
};

BENCHMARK_F(CompressibleFlowOperatorFixture, Test)
(benchmark::State &state)
{
  boost::container::small_vector<double, 12> vec;
  for (auto _ : state)
    {
      this->data->brinkman_obstacle_force->add_load_to_obstacles(*this->data->obstacle_field);
    }
}

MPDG_BENCHMARK_MPI_MAIN;