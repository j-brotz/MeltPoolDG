/**
 * @brief Benchmark for the explicit operator of the compressible flow solver. The benchmarks in
 * this file benchmark the apply_operator() method for two and three space dimensions as well as
 * viscous and non-viscous fluids.
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

template <int dim, typename number, bool is_viscous>
class KernelFixture : public benchmark::Fixture
{
public:
  KernelFixture()
  {}

  // prevent compiler hiding warning (-Woverloaded-virtual)
  using benchmark::Fixture::SetUp;
  using benchmark::Fixture::TearDown;

  void
  SetUp(benchmark::State &) override
  {
    set_parameters();

    scratch_data =
      std::make_unique<MeltPoolDG::ScratchData<dim, dim, number>>(MPI_COMM_WORLD, 0, true);

    tria = std::make_unique<dealii::parallel::distributed::Triangulation<dim>>(MPI_COMM_WORLD);
    dealii::GridGenerator::subdivided_hyper_cube(*tria, 5);
    dof_handler.reinit(*tria);

    MeltPoolDG::FiniteElementUtils::distribute_dofs<dim, dim + 2>(fe, dof_handler);

    // setup scratch data
    scratch_data->set_mapping(MeltPoolDG::FiniteElementUtils::create_mapping<dim>(fe));

    unsigned int quad_index =
      scratch_data->attach_quadrature(MeltPoolDG::FiniteElementUtils::create_quadrature<dim>(fe));

    unsigned int dof_index = scratch_data->attach_dof_handler(dof_handler);

    scratch_data->attach_constraint_matrix(dummy_constraints);

    scratch_data->create_partitioning();

    scratch_data->build(true, true, false, false);

    flow_scratch_data =
      std::make_unique<MeltPoolDG::Flow::CompressibleFlowScratchData<dim, number>>(flow_data,
                                                                                   *scratch_data,
                                                                                   dof_index,
                                                                                   quad_index);

    flow_scratch_data->reinit(1);

    set_boundary_conditions();

    set_initial_condition();

    dg_operator = std::make_unique<
      MeltPoolDG::Flow::DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>>(
      *flow_scratch_data);
  }

  void
  TearDown(benchmark::State &) override
  {
    scratch_data->clear();
    scratch_data.reset();
    flow_scratch_data.reset();
    tria.reset();
  }


  std::unique_ptr<MeltPoolDG::Flow::CompressibleFlowScratchData<dim, number>> flow_scratch_data;
  std::unique_ptr<MeltPoolDG::Flow::DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>>
    dg_operator;

private:
  void
  set_initial_condition()
  {
    InflowFlowField<dim, number>                   function(0.);
    dealii::FECellIntegrator<dim, dim + 2, number> phi(
      flow_scratch_data->scratch_data.get_matrix_free(),
      flow_scratch_data->dof_idx,
      flow_scratch_data->quad_idx);
    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, number> inverse(phi);
    flow_scratch_data->solution_history.get_current_solution().zero_out_ghost_values();
    for (unsigned int cell = 0;
         cell < flow_scratch_data->scratch_data.get_matrix_free().n_cell_batches();
         ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
          phi.submit_dof_value(
            MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
              function, phi.quadrature_point(q)),
            q);
        inverse.transform_from_q_points_to_basis(dim + 2,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(flow_scratch_data->solution_history.get_current_solution());
      }
  }

  void
  set_parameters()
  {
    // finite element data
    fe.type   = MeltPoolDG::FiniteElementType::FE_DGQ;
    fe.degree = 2;
  }

  void
  set_boundary_conditions()
  {
    std::shared_ptr<InflowFlowField<dim, number>> func =
      std::make_shared<InflowFlowField<dim, number>>(0.);
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> inflow{
      {0, func}};
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> outflow{
      {1, func}};
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> top{
      {2, func}};
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> bottom{
      {3, func}};
    flow_scratch_data->boundary_conditions.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::inflow, inflow);
    flow_scratch_data->boundary_conditions.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::subsonic_outflow_fixed_energy, outflow);
    flow_scratch_data->boundary_conditions.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::no_slip_wall, top);
    flow_scratch_data->boundary_conditions.set_boundary_condition(
      MeltPoolDG::Flow::CompressibleBoundaryConditionType::no_slip_wall, bottom);
  }

  std::unique_ptr<dealii::Triangulation<dim>>                tria;
  MeltPoolDG::Flow::CompressibleFlowData<number>             flow_data;
  std::unique_ptr<MeltPoolDG::ScratchData<dim, dim, number>> scratch_data;
  dealii::DoFHandler<dim>                                    dof_handler;
  MeltPoolDG::FiniteElementData                              fe;
  dealii::AffineConstraints<number>                          dummy_constraints;
};

BENCHMARK_TEMPLATE_F(KernelFixture, ExplicitOperatorViscousDim2, 2, double, true)(
  benchmark::State &st)
{
  for (auto _ : st)
    {
      dg_operator->apply_operator(0.,
                                  flow_scratch_data->solution_history.get_current_solution(),
                                  flow_scratch_data->solution_history.get_current_solution(),
                                  std::function<void(unsigned int, unsigned int)>());
    }
}

BENCHMARK_TEMPLATE_F(KernelFixture, ExplicitOperatorViscousDim3, 3, double, true)(
  benchmark::State &st)
{
  for (auto _ : st)
    {
      dg_operator->apply_operator(0.,
                                  flow_scratch_data->solution_history.get_current_solution(),
                                  flow_scratch_data->solution_history.get_current_solution(),
                                  std::function<void(unsigned int, unsigned int)>());
    }
}

BENCHMARK_TEMPLATE_F(KernelFixture, ExplicitOperatorEulerDim2, 2, double, false)(
  benchmark::State &st)
{
  for (auto _ : st)
    {
      dg_operator->apply_operator(0.,
                                  flow_scratch_data->solution_history.get_current_solution(),
                                  flow_scratch_data->solution_history.get_current_solution(),
                                  std::function<void(unsigned int, unsigned int)>());
    }
}

BENCHMARK_TEMPLATE_F(KernelFixture, ExplicitOperatorEulerDim3, 3, double, false)(
  benchmark::State &st)
{
  for (auto _ : st)
    {
      dg_operator->apply_operator(0.,
                                  flow_scratch_data->solution_history.get_current_solution(),
                                  flow_scratch_data->solution_history.get_current_solution(),
                                  std::function<void(unsigned int, unsigned int)>());
    }
}

MPDG_BENCHMARK_MPI_MAIN