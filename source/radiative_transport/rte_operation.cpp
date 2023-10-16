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
  RadiativeTransportOperation<dim>::solve()
  {
    ScopedName         sc("rte::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

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
                  [&]() -> double { return solution_history.get_current_solution().l2_norm(); },
                  "pseudo-time-solution",
                  "RTE::pseudo-time-stepping",
                  6 /*precision*/,
                  "l2");
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
              [&]() -> double { return solution_history.get_current_solution().l2_norm(); },
              "intensity",
              "RTE::pseudo-predictor",
              6 /*precision*/,
              "l2");
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

        heaviside.zero_out_ghost_values();

        scratch_data.get_constraint(rte_dof_idx)
          .distribute(solution_history.get_current_solution());

        solution_history.commit_old_solutions();

        const ConditionalOStream &pcout = scratch_data.get_pcout(rte_data.verbosity_level);
        Journal::print_formatted_norm(
          pcout,
          [&]() -> double {
            return VectorTools::compute_L2_norm<dim>(solution_history.get_current_solution(),
                                                     scratch_data,
                                                     rte_dof_idx,
                                                     rte_quad_idx);
          },
          "intensity",
          "RTE",
          11 /*precision*/
        );
      }
    IterationMonitor::add_linear_iterations(sc, iter);
  }
  template <int dim>
  void
  RadiativeTransportOperation<dim>::pseudo_solve()
  {
    if (!solution_history.get_recent_old_solution().has_ghost_elements())
      solution_history.get_recent_old_solution().update_ghost_values();
    if (!heaviside.has_ghost_elements())
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

    if (!solution_history.get_recent_old_solution().has_ghost_elements())
      solution_history.get_recent_old_solution().zero_out_ghost_values();
    if (!heaviside.has_ghost_elements())
      heaviside.zero_out_ghost_values();
    solution_history.get_current_solution().update_ghost_values();

    scratch_data.get_constraint(rte_dof_idx).distribute(solution_history.get_current_solution());
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
