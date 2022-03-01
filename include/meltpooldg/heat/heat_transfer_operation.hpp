/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/heat_transfer_operator.hpp>
#include <meltpooldg/heat/heat_transfer_preconditioner_matrixfree.hpp>
#include <meltpooldg/interface/boundary_conditions.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class HeatTransferOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const ScratchData<dim>                  &scratch_data;
    std::shared_ptr<BoundaryConditions<dim>> bc_data;
    /**
     * parameters
     */
    const HeatData<double>      heat_data;
    const TimeIterator<double> &time_iterator;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int temp_dof_idx;
    const unsigned int temp_hanging_nodes_dof_idx;
    const unsigned int temp_quad_idx;
    /*
     *    This are the primary solution variables of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType temperature_extrapolated;
    VectorType heat_source;
    VectorType user_rhs;
    VectorType temp;
    VectorType temperature_interface;

    // for output only
    mutable VectorType user_rhs_projected;

    std::unique_ptr<Predictor<VectorType, double>> predictor;

    // optional flow velocity for internal convection
    const unsigned int vel_dof_idx;
    const VectorType  *velocity;

    // optional level-set heaviside field for two phase flow
    const unsigned int ls_dof_idx;
    const VectorType  *level_set_as_heaviside;

    NewtonRaphsonSolver<dim> newton;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::shared_ptr<HeatTransferOperator<dim>> heat_operator;

    std::shared_ptr<HeatTransferPreconditionerMatrixFree<dim>> heat_transfer_preconditioner;
    std::shared_ptr<DiagonalMatrix<VectorType>>                diag_preconditioner;
    std::shared_ptr<TrilinosWrappers::PreconditionBase>        trilinos_preconditioner;

    // determine whether solution vectors are prepared for time advance
    bool ready_for_time_advance = false;

    std::unique_ptr<LevelSet::Tools::NearestPoint<dim>> nearest_point_search;

  public:
    HeatTransferOperation(std::shared_ptr<BoundaryConditions<dim>> bc_data,
                          const ScratchData<dim>                  &scratch_data_in,
                          const HeatData<double>                  &heat_data_in,
                          const Material<double>                  &material,
                          const TimeIterator<double>              &time_iterator,
                          unsigned int                             temp_dof_idx_in,
                          unsigned int                             temp_hanging_nodes_dof_idx_in,
                          unsigned int                             temp_quad_idx_in,
                          unsigned int                             vel_dof_idx_in = 0,
                          const VectorType                        *velocity_in    = nullptr,
                          unsigned int                             ls_dof_idx_in  = 0,
                          const VectorType *level_set_as_heaviside_in             = nullptr,
                          const bool        do_solidifiaction                     = false);

    void
    register_evaporative_mass_flux(
      VectorType        *evaporative_mass_flux_in,
      const unsigned int evapor_mass_flux_dof_idx_in,
      const double       latent_heat_of_evaporation,
      const typename Evaporation::EvaporationData<double>::EvaporativeCooling &evapor_cooling_data);

    void
    register_surface_mesh(
      const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                   std::vector<Point<dim>> /*quad_points*/,
                                   std::vector<double> /*weights*/
                                   >> &surface_mesh_info_in);

    void
    set_initial_condition(const Function<dim> &initial_field_function_temperature,
                          const double         start_time);

    void
    reinit();

    void
    init_time_advance();

    void
    solve(const bool do_finish_time_step = true);

    void
    finish_time_advance();

    void
    compute_interface_temperature(const VectorType                         &distance,
                                  const BlockVectorType                    &normal_vector,
                                  const LevelSet::NearestPointData<double> &nearest_point_data);

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

    void
    distribute_constraints();

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    void
    attach_output_vectors_failed_step(GenericDataOut<dim> &data_out) const;

    const VectorType &
    get_temperature() const;

    VectorType &
    get_temperature();

    const VectorType &
    get_temperature_interface() const;

    VectorType &
    get_temperature_interface();

    const VectorType &
    get_heat_source() const;

    VectorType &
    get_heat_source();

    const VectorType &
    get_user_rhs() const;

    VectorType &
    get_user_rhs();

  private:
    void
    setup_newton();
  };
} // namespace MeltPoolDG::Heat
