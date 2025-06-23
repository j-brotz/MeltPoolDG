#include <meltpooldg/level_set/normal_vector_operation.hpp>
#include <meltpooldg/level_set/normal_vector_operator.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  NormalVectorOperation<dim, number>::NormalVectorOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const NormalVectorData<number>      &normal_vector_data,
    const VectorType                    &solution_level_set,
    const std::array<unsigned int, dim> &normal_dof_indices_per_block_in,
    const unsigned int                   normal_no_bc_dof_idx_in,
    const unsigned int                   normal_quad_idx_in,
    const unsigned int                   ls_dof_idx_in)
    : scratch_data(scratch_data_in)
    , normal_vector_data(normal_vector_data)
    , solution_level_set(solution_level_set)
    , normal_no_bc_dof_idx(normal_no_bc_dof_idx_in)
    , normal_dof_indices_per_block(normal_dof_indices_per_block_in)
    , normal_quad_idx(normal_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , solution_history(normal_vector_data.predictor.n_old_solution_vectors)
  {
    if (!normal_vector_operator)
      create_operator();
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::reinit()
  {
    solution_history.apply([this](BlockVectorType &v) {
      scratch_data.initialize_dof_vector(v, normal_dof_indices_per_block);
    });

    scratch_data.initialize_dof_vector(solution_normal_vector_predictor,
                                       normal_dof_indices_per_block);
    scratch_data.initialize_dof_vector(rhs, normal_dof_indices_per_block);

    normal_vector_operator->reinit();

    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        // precompute preconditioner
        const bool update_ghosts = !solution_history.get_current_solution().has_ghost_elements();
        if (update_ghosts)
          solution_history.get_current_solution().update_ghost_values();

        preconditioner.reinit(scratch_data, normal_dof_indices_per_block[0]);
        preconditioner.update();

        if (update_ghosts)
          solution_history.get_current_solution().zero_out_ghost_values();
      }
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::solve()
  {
    const ScopedName   sc("normal::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    const bool update_ghosts = !solution_level_set.has_ghost_elements();
    if (update_ghosts)
      solution_level_set.update_ghost_values();

    // compute predictor
    if (!predictor)
      predictor = std::make_unique<Predictor<BlockVectorType, number>>(normal_vector_data.predictor,
                                                                       solution_history);

    if (normal_vector_data.linear_solver.do_matrix_free &&
        normal_vector_data.predictor.type == PredictorType::least_squares_projection)
      {
        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree<dim, number>(
          *normal_vector_operator,
          rhs,
          solution_level_set,
          scratch_data,
          normal_dof_indices_per_block,
          normal_no_bc_dof_idx,
          true);
      }

    predictor->vmult(*normal_vector_operator, solution_normal_vector_predictor, rhs);

    for (unsigned int d = 0; d < dim; ++d)
      scratch_data.get_constraint(normal_dof_indices_per_block[d])
        .distribute(solution_history.get_current_solution().block(d));

    unsigned int iter = 0;

    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        //  Apply wetting boundary conditions
        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree<dim, number>(
          *normal_vector_operator,
          rhs,
          solution_level_set,
          scratch_data,
          normal_dof_indices_per_block,
          normal_no_bc_dof_idx,
          true);

        iter = LinearSolver::solve<BlockVectorType>(*normal_vector_operator,
                                                    solution_history.get_current_solution(),
                                                    rhs,
                                                    normal_vector_data.linear_solver,
                                                    preconditioner,
                                                    "normal_vector_operation");
      }
    else
      {
        normal_vector_operator->compute_system_matrix_and_rhs(solution_level_set, rhs);

        for (unsigned int d = 0; d < dim; ++d)
          iter = LinearSolver::solve<VectorType>(normal_vector_operator->get_system_matrix(),
                                                 solution_history.get_current_solution().block(d),
                                                 rhs.block(d),
                                                 normal_vector_data.linear_solver,
                                                 PreconditionIdentity(),
                                                 "normal_vector_operation");
      }

    if (update_ghosts)
      solution_level_set.zero_out_ghost_values();

    for (unsigned int d = 0; d < dim; ++d)
      {
        scratch_data.get_constraint(normal_dof_indices_per_block[d])
          .distribute(solution_history.get_current_solution().block(d));
      }
    constexpr int             verbosity_l2_norm = dim > 1 ? 1 : 2;
    const ConditionalOStream &pcout =
      scratch_data.get_pcout(std::max(normal_vector_data.verbosity_level, verbosity_l2_norm));

    for (unsigned int d = 0; d < dim; ++d)
      Journal::print_formatted_norm<number>(
        pcout,
        [&]() -> number {
          return MeltPoolDG::VectorTools::compute_norm<dim, number>(
            solution_history.get_current_solution().block(d),
            scratch_data,
            normal_dof_indices_per_block[d],
            normal_quad_idx);
        },
        "normal_" + std::to_string(d),
        "normal_vector",
        11 /*precision*/
      );

    Journal::print_line(scratch_data.get_pcout(2),
                        "     * CG: i = " + std::to_string(iter),
                        "normal_vector");

    IterationMonitor<number>::add_linear_iterations(sc, iter);

    // update ghost_values of solution
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim, typename number>
  const typename NormalVectorOperation<dim, number>::BlockVectorType &
  NormalVectorOperation<dim, number>::get_solution_normal_vector() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  typename NormalVectorOperation<dim, number>::BlockVectorType &
  NormalVectorOperation<dim, number>::get_solution_normal_vector()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    solution_history.apply([&](BlockVectorType &v) {
      for (unsigned int d = 0; d < dim; ++d)
        vectors.push_back(&v.block(d));
    });
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::create_operator()
  {
    normal_vector_operator =
      std::make_unique<NormalVectorOperator<dim, number>>(scratch_data,
                                                          normal_vector_data,
                                                          normal_dof_indices_per_block,
                                                          normal_quad_idx,
                                                          ls_dof_idx,
                                                          &solution_level_set);

    preconditioner =
      make_preconditioner<dim, number, NormalVectorOperator<dim, number>, BlockVectorType>(
        normal_vector_data.linear_solver.preconditioner_type,
        normal_vector_operator.get(),
        normal_vector_data.linear_solver.do_matrix_free);
  }

  template class NormalVectorOperation<1, double>;
  template class NormalVectorOperation<2, double>;
  template class NormalVectorOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
