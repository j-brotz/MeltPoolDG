#include <deal.II/numerics/data_out.h>

#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG
{
  template <int dim, typename VectorType>
  NewtonRaphsonSolver<dim, VectorType>::NewtonRaphsonSolver(
    const ScratchData<dim> &                    scratch_data,
    const NonlinearSolverData<double> &         nlsolve_data,
    const unsigned int                          dof_idx,
    const unsigned int                          quad_idx,
    const VectorType &                          solution_old,
    VectorType &                                solution,
    const std::function<void(VectorType &rhs)> &create_rhs,
    const std::function<int(VectorType &solution_update, const VectorType &rhs)>
      &solve_linear_system)
    : scratch_data(scratch_data)
    , nlsolve_data(nlsolve_data)
    , dof_idx(dof_idx)
    , quad_idx(quad_idx)
    , solution_old(solution_old)
    , solution(solution)
    , max_number_of_iterations(nlsolve_data.max_nonlinear_iterations +
                               nlsolve_data.max_nonlinear_iterations_alt)
    , residual_tolerance(nlsolve_data.residual_tolerance)
    , field_correction_tolerance(nlsolve_data.field_correction_tolerance)
    , create_rhs(create_rhs)
    , solve_linear_system(solve_linear_system)
  {
    scratch_data.initialize_dof_vector(rhs, dof_idx);
    scratch_data.initialize_dof_vector(solution_update, dof_idx);
  }

  template <int dim, typename VectorType>
  void
  NewtonRaphsonSolver<dim, VectorType>::solve()
  {
    solution.zero_out_ghost_values();

    Journal::print_line(scratch_data.get_pcout(1));
    Journal::print_line(scratch_data.get_pcout(1), std::string(10, ' ') + std::string(60, '_'));
    std::ostringstream str;
    str << std::string(10, ' ') << std::setw(15) << "#lin solve" << std::internal << std::setw(15)
        << "||residual||" << std::internal << std::setw(15) << "||ddx||";
    Journal::print_line(scratch_data.get_pcout(1), str.str(), "newton_raphson_solver");
    Journal::print_line(scratch_data.get_pcout(1), std::string(10, ' ') + std::string(60, '_'));

    int i = 0;
    while (i < max_number_of_iterations)
      {
        solve_increment();

        if (is_converged())
          {
            std::ostringstream str_sol;
            str_sol << "Newton Raphson solver converged: ||solution|| = " << std::scientific
                    << std::setprecision(5)
                    << MeltPoolDG::VectorTools::compute_L2_norm<dim>(solution,
                                                                     scratch_data,
                                                                     dof_idx,
                                                                     quad_idx);

            Journal::print_line(scratch_data.get_pcout(0), str_sol.str(), "newton_raphson_solver");
            return;
          }

        solution += solution_update;
        scratch_data.get_constraint(dof_idx).distribute(solution);

        i++;
      }
    // @todo: make exception
    //
    if (!is_converged())
      {
        DataOut<dim> data_out;

        data_out.add_data_vector(scratch_data.get_dof_handler(dof_idx), solution, "solution");
        data_out.add_data_vector(scratch_data.get_dof_handler(dof_idx),
                                 solution_update,
                                 "solution_update");
        data_out.add_data_vector(scratch_data.get_dof_handler(dof_idx), rhs, "rhs");
        data_out.build_patches(scratch_data.get_mapping());
        data_out.write_vtu_in_parallel("newton_raphson_failed.vtu", scratch_data.get_mpi_comm());
      }

    AssertThrow(is_converged(), ExcMessage("Newton Raphson solver did not converge!"));
  }

  template <int dim, typename VectorType>
  double
  NewtonRaphsonSolver<dim, VectorType>::suggest_new_time_increment()
  {
    AssertThrow(false, ExcNotImplemented());
    return 0.0;
  }

  template <int dim, typename VectorType>
  void
  NewtonRaphsonSolver<dim, VectorType>::set_tolerances_to_alternative_values()
  {
    residual_tolerance         = nlsolve_data.residual_tolerance_alt;
    field_correction_tolerance = nlsolve_data.field_correction_tolerance_alt;
  }

  template <int dim, typename VectorType>
  bool
  NewtonRaphsonSolver<dim, VectorType>::is_converged()
  {
    if (iteration_counter == nlsolve_data.max_nonlinear_iterations)
      set_tolerances_to_alternative_values();

    double res_norm    = rhs.l2_norm();
    double update_norm = solution_update.l2_norm();

    bool residual_converged   = res_norm < residual_tolerance;
    bool correction_converged = update_norm < field_correction_tolerance;

    str_ << std::right << std::setw(15) << std::scientific << std::setprecision(5) << res_norm
         << print_checkmark(residual_converged);
    str_ << std::right << std::setw(15) << std::scientific << std::setprecision(5) << update_norm
         << print_checkmark(correction_converged);

    Journal::print_line(scratch_data.get_pcout(1), str_.str(), "", 4);

    str_.str("");

    return residual_converged && correction_converged;
  }

  template <int dim, typename VectorType>
  std::string
  NewtonRaphsonSolver<dim, VectorType>::print_checkmark(bool is_converged)
  {
    if (is_converged)
      return " ✓ ";
    else
      return " ✗ ";
  }

  template <int dim, typename VectorType>
  void
  NewtonRaphsonSolver<dim, VectorType>::solve_increment()
  {
    rhs             = 0.0;
    solution_update = 0.0;

    create_rhs(rhs);
    int iter = solve_linear_system(solution_update, rhs);
    str_ << std::string(10, ' ') << std::right << std::setw(15) << std::setprecision(0) << iter;
  }

  template class NewtonRaphsonSolver<1>;
  template class NewtonRaphsonSolver<2>;
  template class NewtonRaphsonSolver<3>;
} // namespace MeltPoolDG
