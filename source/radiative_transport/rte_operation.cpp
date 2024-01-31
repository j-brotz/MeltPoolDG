#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/rte_operation.hpp>
#include <meltpooldg/radiative_transport/rte_problem.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>



namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim>
  RadiativeTransportOperation<dim>::RadiativeTransportOperation(
    const ScratchData<dim>               &scratch_data_in,
    const RadiativeTransportData<double> &rte_data_in,
    const VectorType                     &heaviside_in,
    const unsigned int                    rte_dof_idx_in,
    const unsigned int                    rte_hanging_nodes_dof_idx_in,
    const unsigned int                    rte_quad_idx_in,
    const unsigned int                    hs_dof_idx_in)
    : scratch_data(scratch_data_in)
    , rte_data(rte_data_in)
    , heaviside(heaviside_in)
    , rte_dof_idx(rte_dof_idx_in)
    , rte_hanging_nodes_dof_idx(rte_hanging_nodes_dof_idx_in)
    , rte_quad_idx(rte_quad_idx_in)
    , hs_dof_idx(hs_dof_idx_in)
    , solution_history((rte_data.problem_type == RTEProblemType::plain)                  ? 1 :
                       (rte_data.problem_type == RTEProblemType::time_dependent_problem) ? 2 :
                                                                                           3)
    , pseudo_time_iterator(pseudo_time_stepping)
  {
    // TODO: get from `LaserBase`
    for (unsigned int i = 0; i < dim; i++)
      {
        laser_direction[i] = rte_data.laser_direction[i];
      }
    AssertThrow(laser_direction.norm() > 1e-16,
                ExcZero("laser direction has zero norm. Please check .json input parameter file"));
    laser_direction /= laser_direction.norm(); // normalize

    // matrix-based simulation is not supported
    AssertThrow(rte_data.linear_solver.do_matrix_free &&
                  rte_data.pseudo_time_stepping.linear_solver.do_matrix_free,
                ExcNotImplemented("This simulation only supports matrix-free operations."));

    /*
     * operator init and setup preconditioner for matrix-free computation
     */
    if (rte_data.problem_type != RTEProblemType::time_dependent_problem)
      {
        rte_operator = std::make_unique<RadiativeTransportOperator<dim, double>>(
          scratch_data, rte_data, heaviside, rte_dof_idx, rte_quad_idx, hs_dof_idx);
        preconditioner_matrixfree = std::make_shared<
          Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
          scratch_data, rte_dof_idx, rte_data.linear_solver.preconditioner_type, *rte_operator);
      }
    if (rte_data.problem_type != RTEProblemType::plain)
      {
        pseudo_rte_operator = std::make_unique<PseudoRTEOperator<dim, double>>(
          scratch_data, rte_data, heaviside, rte_dof_idx, rte_quad_idx, hs_dof_idx);
        pseudo_preconditioner_matrixfree = std::make_shared<
          Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
          scratch_data,
          rte_dof_idx,
          rte_data.pseudo_time_stepping.linear_solver.preconditioner_type,
          *pseudo_rte_operator);
      }

    reinit();
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::reinit()
  {
    {
      ScopedName sc("rte::n_dofs");
      DoFMonitor::add_n_dofs(sc, scratch_data.get_dof_handler(rte_dof_idx).n_dofs());
    }
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, rte_dof_idx); });

    scratch_data.initialize_dof_vector(rhs, rte_dof_idx);

    if (rte_data.problem_type != RTEProblemType::time_dependent_problem)
      preconditioner_matrixfree->reinit();
    if (rte_data.problem_type != RTEProblemType::plain)
      {
        pseudo_preconditioner_matrixfree->reinit();
        // make pseudo-dt for pseudo-rte operator to access
        if (rte_data.pseudo_time_stepping.time_step_size > 1e-16)
          pseudo_rte_operator->reset_time_increment(rte_data.pseudo_time_stepping.time_step_size);
        else
          pseudo_rte_operator->reset_time_increment(
            this->scratch_data.get_min_cell_size() *
            rte_data.pseudo_time_stepping.pseudo_time_scaling);
      }
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::distribute_constraints()
  {
    scratch_data.get_constraint(rte_dof_idx).distribute(solution_history.get_current_solution());
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::setup_constraints(
    ScratchData<dim>                       &scratch_data_in,
    const DirichletBoundaryConditions<dim> &bc_data,
    const PeriodicBoundaryConditions<dim>  &pbc,
    const unsigned int                      rte_dof_idx_in,
    const unsigned int                      rte_hanging_nodes_idx,
    const bool                              set_inhomogeneities)

  {
    // setup hanging constraints
    scratch_data_in.get_constraint(rte_hanging_nodes_idx).clear();
    scratch_data_in.get_constraint(rte_hanging_nodes_idx)
      .reinit(scratch_data_in.get_locally_relevant_dofs(rte_hanging_nodes_idx));
    DoFTools::make_hanging_node_constraints(scratch_data_in.get_dof_handler(rte_hanging_nodes_idx),
                                            scratch_data_in.get_constraint(rte_hanging_nodes_idx));

    for (const auto &bc : pbc.get_data())
      {
        const auto [id_in, id_out, direction] = bc;
        DoFTools::make_periodicity_constraints(
          scratch_data_in.get_dof_handler(rte_hanging_nodes_idx),
          id_in,
          id_out,
          direction,
          scratch_data_in.get_constraint(rte_hanging_nodes_idx));
      }

    scratch_data_in.get_constraint(rte_hanging_nodes_idx).close();

    UtilityFunctions::check_constraints(scratch_data_in.get_dof_handler(rte_hanging_nodes_idx),
                                        scratch_data_in.get_constraint(rte_hanging_nodes_idx));

    // setup Dirichlet constraints and merge
    scratch_data_in.get_constraint(rte_dof_idx_in).clear();
    scratch_data_in.get_constraint(rte_dof_idx_in)
      .reinit(scratch_data_in.get_locally_relevant_dofs(rte_dof_idx_in));

    if (!bc_data.get_data().empty())
      {
        for (const auto &bc : bc_data.get_data())
          {
            if (set_inhomogeneities)
              dealii::VectorTools::interpolate_boundary_values(
                scratch_data_in.get_mapping(),
                scratch_data_in.get_dof_handler(rte_dof_idx_in),
                bc.first,
                *bc.second,
                scratch_data_in.get_constraint(rte_dof_idx_in));
            else
              dealii::DoFTools::make_zero_boundary_constraints(
                scratch_data_in.get_dof_handler(rte_dof_idx_in),
                bc.first,
                scratch_data_in.get_constraint(rte_dof_idx_in));
          }
      }

    scratch_data_in.get_constraint(rte_dof_idx_in).close();
    UtilityFunctions::check_constraints(scratch_data_in.get_dof_handler(rte_dof_idx_in),
                                        scratch_data_in.get_constraint(rte_dof_idx_in));

    scratch_data_in.get_constraint(rte_dof_idx_in)
      .merge(scratch_data_in.get_constraint(rte_hanging_nodes_idx),
             AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
    scratch_data_in.get_constraint(rte_dof_idx_in).close();
    UtilityFunctions::check_constraints(scratch_data_in.get_dof_handler(rte_dof_idx_in),
                                        scratch_data_in.get_constraint(rte_dof_idx_in));
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

    // 1) Perform pseudo-time stepping to compute an initial guess (predictor) to then solve the
    // radiative transfer equation
    if (rte_data.problem_type != RTEProblemType::plain)
      {
        pseudo_time_iterator.reset();

        // relative change and max_n_steps are used for early stopping,
        // disabled for time_dependent_problem
        if (rte_data.problem_type !=
            RTEProblemType::time_dependent_problem) // early stopping: disabled if the problem is
                                                    // purely pseudo-time dependent
          pseudo_time_iterator.reset_max_n_time_steps(rte_data.pseudo_time_stepping.max_n_steps);
        else
          pseudo_time_iterator.reset_max_n_time_steps(
            100); // default dt is 0.01. This makes pseudo time be sub-interval time
                  // (100 steps = 1s)
        double pseudo_rel_change = rte_data.pseudo_time_stepping.rel_tolerance;

        while (!pseudo_time_iterator.is_finished() &&
               ((rte_data.problem_type == RTEProblemType::time_dependent_predictor &&
                 pseudo_rel_change >= rte_data.pseudo_time_stepping.rel_tolerance) ||
                rte_data.problem_type == RTEProblemType::time_dependent_problem))
          {
            rhs = 0.;
            pseudo_time_iterator.compute_next_time_increment();
            pseudo_solve();
            if (rte_data.problem_type == RTEProblemType::time_dependent_predictor)
              // early-stopping only allowed for a pseudo-time predictor problem
              pseudo_rel_change = std::abs((solution_history.get_current_solution().l2_norm() -
                                            solution_history.get_all_solutions()[2].l2_norm()) /
                                           solution_history.get_all_solutions()[2].l2_norm());

            if (rte_data.problem_type == RTEProblemType::time_dependent_problem)
              {
                Journal::print_formatted_norm(
                  scratch_data.get_pcout(2),
                  [&]() -> double { return pseudo_time_iterator.get_current_time(); },
                  "pseudo-time-out",
                  "RTE::pseudo-time-stepping",
                  1 /*precision*/,
                  "s");
                // print final pseudo-time stepping solution
                Journal::print_formatted_norm(
                  scratch_data.get_pcout(0),
                  [&]() -> double {
                    return VectorTools::compute_norm<dim>(solution_history.get_current_solution(),
                                                          scratch_data,
                                                          rte_dof_idx,
                                                          rte_quad_idx);
                  },
                  "pseudo-time-solution",
                  "RTE::pseudo-time-stepping",
                  6 /*precision*/);
              }

            solution_history.commit_old_solutions();
          }

        // print predictor
        if (rte_data.problem_type == RTEProblemType::time_dependent_predictor)
          {
            Journal::print_formatted_norm(
              scratch_data.get_pcout(2),
              [&]() -> double { return pseudo_time_iterator.get_current_time_step_number(); },
              "n_steps",
              "RTE::pseudo-predictor",
              1 /*precision*/,
              "pseudo-time steps");
            Journal::print_formatted_norm(
              scratch_data.get_pcout(0),
              [&]() -> double {
                return VectorTools::compute_norm<dim>(solution_history.get_current_solution(),
                                                      scratch_data,
                                                      rte_dof_idx,
                                                      rte_quad_idx);
              },
              "intensity",
              "RTE::pseudo-predictor",
              6 /*precision*/);
          }
      }
    // 2) Solve the actual radiative transfer equation
    if (rte_data.problem_type != RTEProblemType::time_dependent_problem)
      {
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
                                                   solution_history.get_current_solution(),
                                                   rhs,
                                                   rte_data.linear_solver,
                                                   *diag_preconditioner_matrixfree);
          }
        else
          {
            trilinos_preconditioner_matrixfree =
              preconditioner_matrixfree->compute_trilinos_preconditioner();
            iter = LinearSolver::solve<VectorType>(*rte_operator,
                                                   solution_history.get_current_solution(),
                                                   rhs,
                                                   rte_data.linear_solver,
                                                   *trilinos_preconditioner_matrixfree);
          }

        if (update_ghosts)
          heaviside.zero_out_ghost_values();

        scratch_data.get_constraint(rte_dof_idx)
          .distribute(solution_history.get_current_solution());

        solution_history.commit_old_solutions();

        Journal::print_formatted_norm(
          scratch_data.get_pcout(0),
          [&]() -> double {
            return VectorTools::compute_norm<dim>(solution_history.get_current_solution(),
                                                  scratch_data,
                                                  rte_dof_idx,
                                                  rte_quad_idx);
          },
          "intensity",
          "RTE",
          11 /*precision*/
        );
      }

    solution_history.get_current_solution().update_ghost_values();

    IterationMonitor::add_linear_iterations(sc, iter);
  }
  template <int dim>
  void
  RadiativeTransportOperation<dim>::pseudo_solve()
  {
    const bool sol_update_ghosts = !solution_history.get_recent_old_solution().has_ghost_elements();
    if (sol_update_ghosts)
      solution_history.get_recent_old_solution().update_ghost_values();
    const bool update_ghosts = !heaviside.has_ghost_elements();
    if (update_ghosts)
      heaviside.update_ghost_values();

    // compute right-hand side of the pseudo-time dependent RTE problem modified by inhomogeneous
    // Dirichlet boundary conditions
    Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree(
      *pseudo_rte_operator,
      rhs,
      solution_history.get_recent_old_solution(),
      scratch_data,
      rte_dof_idx,
      rte_hanging_nodes_dof_idx,
      true);

    if (this->rte_data.pseudo_time_stepping.linear_solver.preconditioner_type ==
        PreconditionerType::Diagonal)
      {
        auto diag_pseudo_preconditioner_matrixfree =
          pseudo_preconditioner_matrixfree->compute_diagonal_preconditioner();

        LinearSolver::solve<VectorType>(*pseudo_rte_operator,
                                        solution_history.get_current_solution(),
                                        rhs,
                                        this->rte_data.pseudo_time_stepping.linear_solver,
                                        *diag_pseudo_preconditioner_matrixfree);
      }
    else
      {
        auto trilinos_pseudo_preconditioner_matrixfree =
          pseudo_preconditioner_matrixfree->compute_trilinos_preconditioner();

        LinearSolver::solve<VectorType>(*pseudo_rte_operator,
                                        solution_history.get_current_solution(),
                                        rhs,
                                        this->rte_data.pseudo_time_stepping.linear_solver,
                                        *trilinos_pseudo_preconditioner_matrixfree);
      }

    if (sol_update_ghosts)
      solution_history.get_recent_old_solution().zero_out_ghost_values();
    if (update_ghosts)
      heaviside.zero_out_ghost_values();

    // update ghost values of relevant solution
    solution_history.get_current_solution().update_ghost_values();

    scratch_data.get_constraint(rte_dof_idx).distribute(solution_history.get_current_solution());
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::compute_heat_source(VectorType        &heat_source,
                                                        const unsigned int heat_source_dof_idx,
                                                        const bool         zero_out) const
  {
    if (zero_out)
      heat_source = 0.0;

    const bool update_ghosts = !solution_history.get_current_solution().has_ghost_elements();
    if (update_ghosts)
      solution_history.get_current_solution().update_ghost_values();

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
            intensity_grad_eval.get_function_gradients(solution_history.get_current_solution(),
                                                       intensity_grad_at_q);

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

    if (update_ghosts)
      solution_history.get_current_solution().zero_out_ghost_values();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  RadiativeTransportOperation<dim>::get_intensity() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  RadiativeTransportOperation<dim>::get_intensity()
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    vectors.push_back(&solution_history.get_current_solution());
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(rte_dof_idx),
                             solution_history.get_current_solution(),
                             "intensity");
    data_out.add_data_vector(scratch_data.get_dof_handler(rte_dof_idx), rhs, "rte_rhs");
  }

  template class RadiativeTransportOperation<1>;
  template class RadiativeTransportOperation<2>;
  template class RadiativeTransportOperation<3>;
} // namespace MeltPoolDG::RadiativeTransport
