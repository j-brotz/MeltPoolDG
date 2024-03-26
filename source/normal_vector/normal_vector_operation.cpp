#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  NormalVectorOperation<dim>::NormalVectorOperation(
    const ScratchData<dim>         &scratch_data_in,
    const NormalVectorData<double> &normal_vector_data,
    const VectorType               &solution_level_set,
    const unsigned int              normal_dof_idx_in,
    const unsigned int              normal_quad_idx_in,
    const unsigned int              ls_dof_idx_in)
    : scratch_data(scratch_data_in)
    , normal_vector_data(normal_vector_data)
    , solution_level_set(solution_level_set)
    , normal_dof_idx(normal_dof_idx_in)
    , normal_quad_idx(normal_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , solution_history(normal_vector_data.predictor.n_old_solution_vectors)
  {
    if (!normal_vector_operator)
      create_operator();
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::reinit()
  {
    solution_history.apply(
      [this](BlockVectorType &v) { scratch_data.initialize_dof_vector(v, normal_dof_idx); });

    scratch_data.initialize_dof_vector(solution_normal_vector_predictor, normal_dof_idx);
    scratch_data.initialize_dof_vector(rhs, normal_dof_idx);

    if (!normal_vector_data.linear_solver.do_matrix_free)
      normal_vector_operator->initialize_matrix_based(scratch_data);

    normal_vector_operator->reinit();

    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        // precompute preconditioner
        const bool update_ghosts = !solution_history.get_current_solution().has_ghost_elements();
        if (update_ghosts)
          solution_history.get_current_solution().update_ghost_values();

        preconditioner_matrixfree->reinit();

        if (normal_vector_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          diag_preconditioner_matrixfree =
            preconditioner_matrixfree->compute_block_diagonal_preconditioner();
        else
          trilinos_preconditioner_matrixfree =
            preconditioner_matrixfree->compute_trilinos_preconditioner();

        if (update_ghosts)
          solution_history.get_current_solution().zero_out_ghost_values();
      }
  }

  template <typename PreconditionerType>
  class BlockPreconditionerWrapper
  {
  public:
    BlockPreconditionerWrapper(const PreconditionerType &precon)
      : precon(precon)
    {}

    template <typename Number>
    void
    vmult(LinearAlgebra::distributed::BlockVector<Number>       &dst,
          const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      for (unsigned int b = 0; b < dst.n_blocks(); ++b)
        precon.vmult(dst.block(b), src.block(b));
    }

  private:
    const PreconditionerType &precon;
  };

  template <int dim>
  void
  NormalVectorOperation<dim>::solve()
  {
    ScopedName         sc("normal::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    const bool update_ghosts = !solution_level_set.has_ghost_elements();
    if (update_ghosts)
      solution_level_set.update_ghost_values();

    // compute predictor
    if (!predictor)
      predictor = std::make_unique<Predictor<BlockVectorType, double>>(normal_vector_data.predictor,
                                                                       solution_history);

    if (normal_vector_data.linear_solver.do_matrix_free &&
        normal_vector_data.predictor.type == PredictorType::least_squares_projection)
      {
        normal_vector_operator->create_rhs(rhs, solution_level_set);
      }

    predictor->vmult(*normal_vector_operator, solution_normal_vector_predictor, rhs);

    // apply hanging node constraints to predictor
    for (unsigned int d = 0; d < dim; ++d)
      scratch_data.get_constraint(normal_dof_idx)
        .distribute(solution_history.get_current_solution().block(d));

    unsigned int iter = 0;

    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        Assert(preconditioner_matrixfree, ExcNotImplemented());

        normal_vector_operator->create_rhs(rhs, solution_level_set);

        if (diag_preconditioner_matrixfree)
          iter = LinearSolver::solve<BlockVectorType>(*normal_vector_operator,
                                                      solution_history.get_current_solution(),
                                                      rhs,
                                                      normal_vector_data.linear_solver,
                                                      *diag_preconditioner_matrixfree,
                                                      "normal_vector_operation");
        else
          iter = LinearSolver::solve<BlockVectorType>(
            *normal_vector_operator,
            solution_history.get_current_solution(),
            rhs,
            normal_vector_data.linear_solver,
            BlockPreconditionerWrapper<TrilinosWrappers::PreconditionBase>(
              *trilinos_preconditioner_matrixfree),
            "normal_vector_operation");
      }
    else
      {
        normal_vector_operator->assemble_matrixbased(solution_level_set,
                                                     normal_vector_operator->get_system_matrix(),
                                                     rhs);

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
        scratch_data.get_constraint(normal_dof_idx)
          .distribute(solution_history.get_current_solution().block(d));
      }
    const unsigned int        verbosity_l2_norm = dim > 1 ? 0 : 1;
    const ConditionalOStream &pcout =
      scratch_data.get_pcout(std::max(normal_vector_data.verbosity_level, verbosity_l2_norm));

    for (unsigned int d = 0; d < dim; ++d)
      Journal::print_formatted_norm(
        pcout,
        [&]() -> double {
          return MeltPoolDG::VectorTools::compute_norm<dim>(
            solution_history.get_current_solution().block(d),
            scratch_data,
            normal_dof_idx,
            normal_quad_idx);
        },
        "normal_" + std::to_string(d),
        "normal_vector",
        11 /*precision*/
      );

    Journal::print_line(scratch_data.get_pcout(1),
                        "     * CG: i = " + std::to_string(iter),
                        "normal_vector");

    IterationMonitor::add_linear_iterations(sc, iter);

    // update ghost_values of solution
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim>
  const typename NormalVectorOperation<dim>::BlockVectorType &
  NormalVectorOperation<dim>::get_solution_normal_vector() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  typename NormalVectorOperation<dim>::BlockVectorType &
  NormalVectorOperation<dim>::get_solution_normal_vector()
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    solution_history.apply([&](BlockVectorType &v) {
      for (unsigned int d = 0; d < dim; ++d)
        vectors.push_back(&v.block(d));
    });
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::create_operator()
  {
    normal_vector_operator = std::make_unique<NormalVectorOperator<dim>>(scratch_data,
                                                                         normal_vector_data,
                                                                         normal_dof_idx,
                                                                         normal_quad_idx,
                                                                         ls_dof_idx,
                                                                         &solution_level_set);
    /*
     * initialize preconditioner matrix-free
     */
    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        preconditioner_matrixfree = std::make_shared<
          Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
          scratch_data,
          normal_dof_idx,
          normal_vector_data.linear_solver.preconditioner_type,
          *normal_vector_operator);
      }
  }


  template class NormalVectorOperation<1>;
  template class NormalVectorOperation<2>;
  template class NormalVectorOperation<3>;
} // namespace MeltPoolDG::LevelSet
