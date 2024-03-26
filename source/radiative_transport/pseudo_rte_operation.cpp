#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/radiative_transport/pseudo_rte_operation.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <limits>


namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim>
  PseudoRTEOperation<dim>::PseudoRTEOperation(const ScratchData<dim>               &scratch_data_in,
                                              const RadiativeTransportData<double> &rte_data_in,
                                              const Tensor<1, dim, double> &laser_direction_in,
                                              const VectorType             &heaviside_in,
                                              const unsigned int            rte_dof_idx_in,
                                              const unsigned int rte_hanging_nodes_dof_idx_in,
                                              const unsigned int rte_quad_idx_in,
                                              const unsigned int hs_dof_idx_in)
    : scratch_data(scratch_data_in)
    , rte_data(rte_data_in)
    , heaviside(heaviside_in)
    , rte_dof_idx(rte_dof_idx_in)
    , rte_hanging_nodes_dof_idx(rte_hanging_nodes_dof_idx_in)
    , rte_quad_idx(rte_quad_idx_in)
    , hs_dof_idx(hs_dof_idx_in)
    , solution_history(2)
    , pseudo_time_iterator(rte_data.pseudo_time_stepping.time_stepping_data)
  {
    pseudo_rte_operator       = std::make_unique<PseudoRTEOperator<dim, double>>(scratch_data,
                                                                           rte_data_in,
                                                                           laser_direction_in,
                                                                           heaviside,
                                                                           rte_dof_idx,
                                                                           rte_quad_idx,
                                                                           hs_dof_idx);
    preconditioner_matrixfree = std::make_shared<
      Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
      scratch_data,
      rte_dof_idx,
      rte_data.pseudo_time_stepping.linear_solver.preconditioner_type,
      *pseudo_rte_operator);
  }

  template <int dim>
  void
  PseudoRTEOperation<dim>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, rte_dof_idx); });

    scratch_data.initialize_dof_vector(rhs, rte_hanging_nodes_dof_idx);

    preconditioner_matrixfree->reinit();

    if (rte_data.pseudo_time_stepping.time_stepping_data.time_step_size > 1e-16)
      pseudo_rte_operator->reset_time_increment(
        rte_data.pseudo_time_stepping.time_stepping_data.time_step_size);
    else
      pseudo_rte_operator->reset_time_increment(this->scratch_data.get_min_cell_size() *
                                                rte_data.pseudo_time_stepping.pseudo_time_scaling);
  }

  template <int dim>
  void
  PseudoRTEOperation<dim>::perform_pseudo_time_stepping()
  {
    pseudo_time_iterator.reset();

    double pseudo_rel_change = std::numeric_limits<double>::max();

    while (!pseudo_time_iterator.is_finished() &&
           pseudo_rel_change >= rte_data.pseudo_time_stepping.rel_tolerance)
      {
        pseudo_time_iterator.compute_next_time_increment();
        solution_history.commit_old_solutions();
        solve();
        const auto recent_l2_norm = solution_history.get_recent_old_solution().l2_norm();
        pseudo_rel_change         = std::abs(
          (solution_history.get_current_solution().l2_norm() - recent_l2_norm) / recent_l2_norm);

        if (rte_data.verbosity_level >= 4)
          Journal::print_formatted_norm(
            scratch_data.get_pcout(0),
            [&]() -> double { return pseudo_time_iterator.get_current_time(); },
            "pseudo-time-out",
            "RTE::pseudo-time-stepping",
            1 /*precision*/,
            "s");
        // print final pseudo-time stepping solution
        if (rte_data.verbosity_level >= 3)
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

    if (rte_data.verbosity_level >= 2)
      Journal::print_formatted_norm(
        scratch_data.get_pcout(0),
        [&]() -> double { return pseudo_time_iterator.get_current_time_step_number(); },
        "n_steps",
        "RTE::pseudo-predictor",
        1 /*precision*/,
        "pseudo-time steps");
    if (rte_data.verbosity_level >= 1)
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


  template <int dim>
  void
  PseudoRTEOperation<dim>::solve()
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
        diag_preconditioner_matrixfree =
          preconditioner_matrixfree->compute_diagonal_preconditioner();

        LinearSolver::solve<VectorType>(*pseudo_rte_operator,
                                        solution_history.get_current_solution(),
                                        rhs,
                                        this->rte_data.pseudo_time_stepping.linear_solver,
                                        *diag_preconditioner_matrixfree,
                                        "pseudo_rte");
      }
    else
      {
        trilinos_preconditioner_matrixfree =
          preconditioner_matrixfree->compute_trilinos_preconditioner();

        LinearSolver::solve<VectorType>(*pseudo_rte_operator,
                                        solution_history.get_current_solution(),
                                        rhs,
                                        this->rte_data.pseudo_time_stepping.linear_solver,
                                        *trilinos_preconditioner_matrixfree,
                                        "pseudo_rte");
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
  PseudoRTEOperation<dim>::set_intensity(
    const LinearAlgebra::distributed::Vector<double> &intensity_in)
  {
    solution_history.get_current_solution() = intensity_in;
  }


  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  PseudoRTEOperation<dim>::get_predicted_intensity() const
  {
    return solution_history.get_current_solution();
  }

  template class PseudoRTEOperation<1>;
  template class PseudoRTEOperation<2>;
  template class PseudoRTEOperation<3>;
} // namespace MeltPoolDG::RadiativeTransport
