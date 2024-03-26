#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/radiative_transport/rte_operation.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>


namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim>
  RadiativeTransportOperation<dim>::RadiativeTransportOperation(
    const ScratchData<dim>               &scratch_data_in,
    const RadiativeTransportData<double> &rte_data_in,
    const Tensor<1, dim, double>         &laser_direction_in,
    const VectorType                     &heaviside_in,
    const unsigned int                    rte_dof_idx_in,
    const unsigned int                    rte_hanging_nodes_dof_idx_in,
    const unsigned int                    rte_quad_idx_in,
    const unsigned int                    hs_dof_idx_in)
    : scratch_data(scratch_data_in)
    , rte_data(rte_data_in)
    , laser_direction(laser_direction_in)
    , heaviside(heaviside_in)
    , rte_dof_idx(rte_dof_idx_in)
    , rte_hanging_nodes_dof_idx(rte_hanging_nodes_dof_idx_in)
    , rte_quad_idx(rte_quad_idx_in)
    , hs_dof_idx(hs_dof_idx_in)
  {
    // matrix-based simulation is not supported
    AssertThrow(rte_data.linear_solver.do_matrix_free &&
                  rte_data.pseudo_time_stepping.linear_solver.do_matrix_free,
                ExcNotImplemented("This simulation only supports matrix-free operations."));

    /*
     * operator init and setup preconditioner for matrix-free computation
     */
    rte_operator = std::make_unique<RadiativeTransportOperator<dim, double>>(
      scratch_data, rte_data, laser_direction, heaviside, rte_dof_idx, rte_quad_idx, hs_dof_idx);
    preconditioner_matrixfree = std::make_shared<
      Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
      scratch_data, rte_dof_idx, rte_data.linear_solver.preconditioner_type, *rte_operator);

    if (rte_data.predictor_type == RTEPredictorType::pseudo_time_stepping)
      pseudo_rte_operation = std::make_unique<PseudoRTEOperation<dim>>(scratch_data,
                                                                       rte_data,
                                                                       laser_direction,
                                                                       heaviside,
                                                                       rte_dof_idx,
                                                                       rte_hanging_nodes_dof_idx,
                                                                       rte_quad_idx,
                                                                       hs_dof_idx);
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::reinit()
  {
    {
      ScopedName sc("rte::n_dofs");
      DoFMonitor::add_n_dofs(sc, scratch_data.get_dof_handler(rte_dof_idx).n_dofs());
    }

    scratch_data.initialize_dof_vector(intensity, rte_dof_idx);
    scratch_data.initialize_dof_vector(rhs, rte_dof_idx);

    preconditioner_matrixfree->reinit();

    if (rte_data.predictor_type == RTEPredictorType::pseudo_time_stepping)
      pseudo_rte_operation->reinit();
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::distribute_constraints()
  {
    scratch_data.get_constraint(rte_dof_idx).distribute(intensity);
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::setup_constraints(
    ScratchData<dim>                       &scratch_data_in,
    const DirichletBoundaryConditions<dim> &bc_data,
    const PeriodicBoundaryConditions<dim>  &pbc)
  {
    Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_BC_into_DBC(
      scratch_data_in, bc_data, pbc, rte_dof_idx, rte_hanging_nodes_dof_idx, true);
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::solve()
  {
    ScopedName         sc("rte::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);
    const bool         update_ghosts = !heaviside.has_ghost_elements();
    if (update_ghosts)
      heaviside.update_ghost_values();

    unsigned int iter = 0;

    // 1) predictor
    if (rte_data.predictor_type == RTEPredictorType::pseudo_time_stepping)
      {
        // Perform pseudo-time stepping to compute an initial guess (predictor)
        pseudo_rte_operation->perform_pseudo_time_stepping();
        intensity = pseudo_rte_operation->get_predicted_intensity();
      }

    // 2) Solve the actual radiative transfer equation
    // apply real dirichlet boundary values
    Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree(*rte_operator,
                                                                     rhs,
                                                                     heaviside,
                                                                     scratch_data,
                                                                     rte_dof_idx,
                                                                     rte_hanging_nodes_dof_idx,
                                                                     true /*zero out rhs*/);

    if (rte_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
      {
        diag_preconditioner_matrixfree =
          preconditioner_matrixfree->compute_diagonal_preconditioner();
        iter = LinearSolver::solve<VectorType>(*rte_operator,
                                               intensity,
                                               rhs,
                                               rte_data.linear_solver,
                                               *diag_preconditioner_matrixfree,
                                               "rte");
      }
    else
      {
        trilinos_preconditioner_matrixfree =
          preconditioner_matrixfree->compute_trilinos_preconditioner();
        iter = LinearSolver::solve<VectorType>(*rte_operator,
                                               intensity,
                                               rhs,
                                               rte_data.linear_solver,
                                               *trilinos_preconditioner_matrixfree,
                                               "rte");
      }

    if (update_ghosts)
      heaviside.zero_out_ghost_values();

    scratch_data.get_constraint(rte_dof_idx).distribute(intensity);

    if (rte_data.predictor_type == RTEPredictorType::pseudo_time_stepping)
      pseudo_rte_operation->set_intensity(intensity);

    Journal::print_formatted_norm(
      scratch_data.get_pcout(0),
      [&]() -> double {
        return VectorTools::compute_norm<dim>(intensity, scratch_data, rte_dof_idx, rte_quad_idx);
      },
      "intensity",
      "RTE",
      11 /*precision*/
    );

    intensity.update_ghost_values();

    IterationMonitor::add_linear_iterations(sc, iter);
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::compute_heat_source(VectorType        &heat_source,
                                                        const unsigned int heat_source_dof_idx,
                                                        const bool         zero_out) const
  {
    if (zero_out)
      heat_source = 0.0;

    const bool update_ghosts = !intensity.has_ghost_elements();
    if (update_ghosts)
      intensity.update_ghost_values();

    // declarations
    FEValues<dim> heat_source_eval(
      scratch_data.get_mapping(),
      scratch_data.get_fe(heat_source_dof_idx),
      Quadrature<dim>(scratch_data.get_fe(heat_source_dof_idx).get_unit_support_points()),
      update_quadrature_points); // dst
    FEValues<dim> intensity_grad_eval(
      scratch_data.get_mapping(),
      scratch_data.get_fe(rte_dof_idx),
      Quadrature<dim>(scratch_data.get_fe(heat_source_dof_idx).get_unit_support_points()),
      update_gradients); // src
    const unsigned int dofs_per_cell = scratch_data.get_fe(heat_source_dof_idx).n_dofs_per_cell();
    std::vector<types::global_dof_index>        local_dof_indices(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim, double>> intensity_grad_at_q(
      intensity_grad_eval.n_quadrature_points);
    VectorType heat_source_multiplicity;
    heat_source_multiplicity.reinit(heat_source);

    for (const auto &cell : scratch_data.get_triangulation().active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            // make iterators
            TriaIterator<DoFCellAccessor<dim, dim, false>> intensity_grad_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(rte_dof_idx));

            TriaIterator<DoFCellAccessor<dim, dim, false>> heat_source_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(heat_source_dof_idx));

            heat_source_dof_cell->get_dof_indices(local_dof_indices);

            // record multiplicity entry
            Vector<double> heat_source_multiplicity_local(dofs_per_cell);
            for (auto &val : heat_source_multiplicity_local)
              val = 1.0;
            scratch_data.get_constraint(heat_source_dof_idx)
              .distribute_local_to_global(heat_source_multiplicity_local,
                                          local_dof_indices,
                                          heat_source_multiplicity);

            // reinit and eval
            heat_source_eval.reinit(heat_source_dof_cell);
            intensity_grad_eval.reinit(intensity_grad_dof_cell);
            intensity_grad_eval.get_function_gradients(intensity, intensity_grad_at_q);

            Vector<double> heat_source_vector_local(dofs_per_cell);

            // get local evaluation
            for (const auto q : heat_source_eval.quadrature_point_indices())
              {
                heat_source_vector_local[q] =
                  std::abs(scalar_product(intensity_grad_at_q[q], laser_direction));
              }
            scratch_data.get_constraint(heat_source_dof_idx)
              .distribute_local_to_global(heat_source_vector_local, local_dof_indices, heat_source);
          }
      }
    heat_source.compress(VectorOperation::add);
    heat_source_multiplicity.compress(VectorOperation::add);

    /*
     * average the heat source added, because an entry is written to multiple times
     */
    for (unsigned int source_mult_local_index = 0;
         source_mult_local_index < heat_source_multiplicity.locally_owned_size();
         ++source_mult_local_index)
      if (heat_source_multiplicity.local_element(source_mult_local_index) > 1.0)
        heat_source.local_element(source_mult_local_index) /=
          heat_source_multiplicity.local_element(source_mult_local_index);

    scratch_data.get_constraint(heat_source_dof_idx).distribute(heat_source);

    if (update_ghosts)
      intensity.zero_out_ghost_values();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  RadiativeTransportOperation<dim>::get_intensity() const
  {
    return intensity;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  RadiativeTransportOperation<dim>::get_intensity()
  {
    return intensity;
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    vectors.push_back(&intensity);
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(rte_dof_idx), intensity, "intensity");
    data_out.add_data_vector(scratch_data.get_dof_handler(rte_dof_idx), rhs, "rte_rhs");
  }

  template class RadiativeTransportOperation<1>;
  template class RadiativeTransportOperation<2>;
  template class RadiativeTransportOperation<3>;
} // namespace MeltPoolDG::RadiativeTransport
