#include <meltpooldg/curvature/curvature_operation.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::Curvature
{
  template <int dim>
  CurvatureOperation<dim>::CurvatureOperation(const ScratchData<dim> &        scratch_data_in,
                                              const CurvatureData<double> &   curvature_data,
                                              const NormalVectorData<double> &normal_vec_data,
                                              const VectorType &              solution_levelset,
                                              const unsigned int              curv_dof_idx_in,
                                              const unsigned int              curv_quad_idx_in,
                                              const unsigned int              normal_dof_idx_in,
                                              const unsigned int              ls_dof_idx_in)
    : scratch_data(scratch_data_in)
    , curvature_data(curvature_data)
    , solution_levelset(solution_levelset)
    , curv_dof_idx(curv_dof_idx_in)
    , curv_quad_idx(curv_quad_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , normal_vector_operation(scratch_data,
                              normal_vec_data,
                              solution_levelset,
                              normal_dof_idx_in,
                              curv_quad_idx,
                              ls_dof_idx_in)
    , solution_history(curvature_data.predictor.n_old_solution_vectors)
  {
    AssertThrow(curvature_data.linear_solver.solver_type == LinearSolverType::CG,
                ExcMessage("The curvature operation only supports the CG solver type."));

    if (!curvature_operator)
      create_operator(solution_levelset);

    reinit();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::update_normal_vector()
  {
    normal_vector_operation.solve();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::solve()
  {
    /*
     *    compute and solve the normal vector field for the given level set
     */
    normal_vector_operation.solve();

    if (!curvature_data.compute_curvature)
      return;

    solution_levelset.update_ghost_values();
    normal_vector_operation.get_solution_normal_vector().update_ghost_values();

    // compute predictor
    if (!predictor)
      predictor =
        std::make_unique<Predictor<VectorType, double>>(curvature_data.predictor, solution_history);

    if (curvature_data.linear_solver.do_matrix_free &&
        curvature_data.predictor.type == PredictorType::least_squares_projection)
      {
        curvature_operator->create_rhs(rhs, normal_vector_operation.get_solution_normal_vector());
      }

    predictor->vmult(*curvature_operator, solution_curvature_predictor, rhs);

    // no need to compute curvature in 1d
    if (dim == 1)
      return;

    unsigned int iter = 0;

    if (curvature_data.linear_solver.do_matrix_free)
      {
        AssertThrow(preconditioner_matrixfree, ExcNotImplemented());

        curvature_operator->create_rhs(rhs, normal_vector_operation.get_solution_normal_vector());

        if (curvature_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          {
            iter = LinearSolver::solve<VectorType>(*curvature_operator,
                                                   solution_history.get_current_solution(),
                                                   rhs,
                                                   curvature_data.linear_solver,
                                                   *diag_preconditioner_matrixfree);
          }
        else
          {
            iter = LinearSolver::solve<VectorType>(*curvature_operator,
                                                   solution_history.get_current_solution(),
                                                   rhs,
                                                   curvature_data.linear_solver,
                                                   *trilinos_preconditioner_matrixfree);
          }
      }
    else
      {
        AssertThrow(
          !curvature_data.do_narrow_band,
          ExcMessage(
            "The computation of the curvature in a narrow band is only implemented matrix-free."));
        curvature_operator->assemble_matrixbased(
          normal_vector_operation.get_solution_normal_vector(),
          curvature_operator->get_system_matrix(),
          rhs);

        iter = LinearSolver::solve<VectorType>(curvature_operator->get_system_matrix(),
                                               solution_history.get_current_solution(),
                                               rhs,
                                               curvature_data.linear_solver);
      }

    solution_levelset.zero_out_ghost_values();
    normal_vector_operation.get_solution_normal_vector().zero_out_ghost_values();

    scratch_data.get_constraint(curv_dof_idx).distribute(solution_history.get_current_solution());

    const unsigned int        verbosity_l2_norm = dim > 1 ? 0 : 1;
    const ConditionalOStream &pcout =
      scratch_data.get_pcout(std::max(curvature_data.verbosity_level, verbosity_l2_norm));


    Journal::print_formatted_norm(
      pcout,
      VectorTools::compute_L2_norm<dim>(
        solution_history.get_current_solution(), scratch_data, curv_dof_idx, curv_quad_idx),
      "curvature",
      "curvature",
      11 /*precision*/
    );

    Journal::print_line(scratch_data.get_pcout(1),
                        "     * CG: i = " + std::to_string(iter),
                        "curvature");
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  CurvatureOperation<dim>::get_curvature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  CurvatureOperation<dim>::get_curvature()
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  const LinearAlgebra::distributed::BlockVector<double> &
  CurvatureOperation<dim>::get_normal_vector() const
  {
    return normal_vector_operation.get_solution_normal_vector();
  }

  template <int dim>
  LinearAlgebra::distributed::BlockVector<double> &
  CurvatureOperation<dim>::get_normal_vector()
  {
    return normal_vector_operation.get_solution_normal_vector();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, curv_dof_idx); });

    scratch_data.initialize_dof_vector(rhs, curv_dof_idx);
    scratch_data.initialize_dof_vector(solution_curvature_predictor, curv_dof_idx);

    if (!curvature_data.linear_solver.do_matrix_free)
      curvature_operator->initialize_matrix_based(scratch_data);

    if (curvature_operator)
      curvature_operator->reinit();

    if (curvature_data.linear_solver.do_matrix_free)
      {
        /*
         * setup sparsity pattern of system matrix only if the latter is
         * needed for computing the preconditioner
         */
        preconditioner_matrixfree->reinit();
        /*
         * precompute system matrix
         */
        if (curvature_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          diag_preconditioner_matrixfree =
            preconditioner_matrixfree->compute_diagonal_preconditioner();
        else
          trilinos_preconditioner_matrixfree =
            preconditioner_matrixfree->compute_trilinos_preconditioner();
      }

    normal_vector_operation.reinit();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    normal_vector_operation.attach_vectors(vectors);

    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim>
  void
  CurvatureOperation<dim>::create_operator(const VectorType &solution_levelset)
  {
    curvature_operator = std::make_shared<CurvatureOperator<dim>>(scratch_data,
                                                                  curvature_data,
                                                                  curv_dof_idx,
                                                                  curv_quad_idx,
                                                                  normal_dof_idx,
                                                                  ls_dof_idx,
                                                                  curvature_data.do_narrow_band,
                                                                  &solution_levelset);
    /*
     *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
     *  apply it to the system matrix. This functionality is part of the OperatorBase class.
     */
    if (!curvature_data.linear_solver.do_matrix_free)
      {
        curvature_operator->initialize_matrix_based(scratch_data);
      }

    curvature_operator->reinit();
    /*
     * initialize preconditioner matrix-free
     */
    if (curvature_data.linear_solver.do_matrix_free)
      {
        preconditioner_matrixfree = std::make_shared<
          Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
          scratch_data,
          curv_dof_idx,
          curvature_data.linear_solver.preconditioner_type,
          *curvature_operator);
        /*
         * precompute system matrix
         */
        if (curvature_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          diag_preconditioner_matrixfree =
            preconditioner_matrixfree->compute_diagonal_preconditioner();
        else
          trilinos_preconditioner_matrixfree =
            preconditioner_matrixfree->compute_trilinos_preconditioner();
      }
  }

  template class CurvatureOperation<1>;
  template class CurvatureOperation<2>;
  template class CurvatureOperation<3>;
} // namespace MeltPoolDG::Curvature
