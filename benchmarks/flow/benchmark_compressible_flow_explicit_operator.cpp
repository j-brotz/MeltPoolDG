/**
 * @brief Benchmark for the explicit operator of the compressible flow solver. The benchmarks in
 * this file benchmarks the apply_operator() method for viscid and in-viscid flows.
 */

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_explicit.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_implicit.hpp>
#include <meltpooldg/utilities/fe_util.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include <memory>
#include <type_traits>

namespace
{
  constexpr int dim = 3;
  using number      = double;

  /**
   * Function used for boundary and initial conditions.
   */
  template <unsigned int dim, typename number>
  class InflowFlowField : public dealii::Function<dim, number>
  {
  public:
    explicit InflowFlowField(const number time)
      : dealii::Function<dim, number>(dim + 2, time)
    {}

    number
    value(const dealii::Point<dim, number> &, const unsigned int component) const final
    {
      if (component == 0)
        return 1.;
      else if (component == 1)
        return 0.4;
      else if (component == dim + 1)
        return 3.097857142857143;
      else
        return 0.;
    }
  };

  template <typename is_viscous_bool>
  class ExplicitOperatorFixture : public benchmark::Fixture
  {
    static const bool is_viscous = is_viscous_bool::value;

  public:
    struct Data
    {
      explicit Data(const benchmark::State &state)
        : triangulation(MPI_COMM_WORLD)
        , scratch_data(MPI_COMM_WORLD, 0, true)
      {
        set_simulation_parameters();
        dealii::GridGenerator::subdivided_hyper_cube(triangulation, state.range(0));
        dof_handler.reinit(triangulation);

        MeltPoolDG::FiniteElementUtils::distribute_dofs<dim, dim + 2>(fe_data, dof_handler);

        setup_scratch_data();

        set_initial_and_boundary_conditions();

        flow_operator = std::make_unique<
          MeltPoolDG::Flow::DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>>(
          *flow_scratch_data);
      }

      dealii::parallel::distributed::Triangulation<dim> triangulation;
      dealii::DoFHandler<dim>                           dof_handler;
      MeltPoolDG::FiniteElementData                     fe_data;
      MeltPoolDG::Flow::CompressibleFlowData<number>    flow_data;
      dealii::AffineConstraints<number>                 affine_constraints;
      MeltPoolDG::ScratchData<dim, dim, number>         scratch_data;

      std::unique_ptr<MeltPoolDG::Flow::CompressibleFlowScratchData<dim, number>> flow_scratch_data;
      std::unique_ptr<MeltPoolDG::Flow::DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>>
        flow_operator;

    private:
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

        flow_scratch_data =
          std::make_unique<MeltPoolDG::Flow::CompressibleFlowScratchData<dim, number>>(flow_data,
                                                                                       scratch_data,
                                                                                       dof_index,
                                                                                       quad_index);
        flow_scratch_data->reinit(1);
      }

      void
      set_simulation_parameters()
      {
        fe_data.type   = MeltPoolDG::FiniteElementType::FE_DGQ;
        fe_data.degree = 2;
      }

      void
      set_initial_and_boundary_conditions()
      {
        // set the initial condition
        InflowFlowField<dim, number>                   function(0.);
        dealii::FECellIntegrator<dim, dim + 2, number> phi(
          flow_scratch_data->scratch_data.get_matrix_free(),
          flow_scratch_data->dof_idx,
          flow_scratch_data->quad_idx);
        dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, number> inverse(
          phi);
        flow_scratch_data->solution_history.get_current_solution().zero_out_ghost_values();
        for (unsigned int cell = 0;
             cell < flow_scratch_data->scratch_data.get_matrix_free().n_cell_batches();
             ++cell)
          {
            phi.reinit(cell);
            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_dof_value(MeltPoolDG::VectorTools::
                                     evaluate_function_at_vectorized_points<dim, number, dim + 2>(
                                       function, phi.quadrature_point(q)),
                                   q);
            inverse.transform_from_q_points_to_basis(dim + 2,
                                                     phi.begin_dof_values(),
                                                     phi.begin_dof_values());
            phi.set_dof_values(flow_scratch_data->solution_history.get_current_solution());
          }

        // set the boundary conditions
        auto func = std::make_shared<InflowFlowField<dim, number>>(0.);
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> inflow{
          {0, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>>
          outflow{{1, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> top{
          {2, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> bottom{
          {3, func}};
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::Flow::CompressibleBoundaryConditionType::inflow, inflow);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::Flow::CompressibleBoundaryConditionType::subsonic_outflow_fixed_energy,
          outflow);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::Flow::CompressibleBoundaryConditionType::no_slip_wall, top);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::Flow::CompressibleBoundaryConditionType::no_slip_wall, bottom);
      }
    };

    // prevent compiler hiding warning (-Woverloaded-virtual)
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

    void
    SetUp(benchmark::State &state) override
    {
      data = std::make_unique<Data>(state);
    }

    void
    TearDown(benchmark::State &) override
    {
      data.reset();
    }

    std::unique_ptr<Data> data;
  };

  BENCHMARK_TEMPLATE_METHOD_F(ExplicitOperatorFixture, ExplicitOperator)(benchmark::State &state)
  {
    for (auto _ : state)
      {
        this->data->flow_operator->apply_operator(
          0.,
          this->data->flow_scratch_data->solution_history.get_current_solution(),
          this->data->flow_scratch_data->solution_history.get_current_solution(),
          std::function<void(unsigned int, unsigned int)>());
      }
    state.counters["DoFs"] = this->data->dof_handler.n_dofs();
  }

  BENCHMARK_TEMPLATE_INSTANTIATE_F(ExplicitOperatorFixture,
                                   ExplicitOperator,
                                   std::integral_constant<bool, true>)
    ->Arg(5)
    ->Arg(10)
    ->Name("viscid");

  BENCHMARK_TEMPLATE_INSTANTIATE_F(ExplicitOperatorFixture,
                                   ExplicitOperator,
                                   std::integral_constant<bool, false>)
    ->Arg(5)
    ->Arg(10)
    ->Name("in-viscid");
} // namespace

MPDG_BENCHMARK_MPI_MAIN;
