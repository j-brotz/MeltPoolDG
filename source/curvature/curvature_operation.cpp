#include <meltpooldg/curvature/curvature_operation.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::Curvature
{
  template <int dim>
  CurvatureOperation<dim>::CurvatureOperation(const ScratchData<dim> &        scratch_data_in,
                                              const CurvatureData<double> &   curvature_data,
                                              const NormalVectorData<double> &normal_vec_data,
                                              const unsigned int              curv_dof_idx_in,
                                              const unsigned int              curv_quad_idx_in,
                                              const unsigned int              normal_dof_idx_in,
                                              const unsigned int              ls_dof_idx_in)
    : scratch_data(scratch_data_in)
    , curvature_data(curvature_data)
    , curv_dof_idx(curv_dof_idx_in)
    , curv_quad_idx(curv_quad_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , normal_vector_operation(scratch_data,
                              normal_vec_data,
                              normal_dof_idx_in,
                              curv_quad_idx,
                              ls_dof_idx_in)
  {
    AssertThrow(curvature_data.linear_solver.solver_type == LinearSolverType::CG,
                ExcMessage("The curvature operation only supports the CG solver type."));
    /*
     *    initialize normal_vector_operation for computing the normal vector to the given
     *    scalar function for which the curvature should be calculated.
     */
  }

  template <int dim>
  void
  CurvatureOperation<dim>::solve(const VectorType &solution_levelset)
  {
    if (!curvature_operator)
      create_operator(solution_levelset);
    /*
     *    compute and solve the normal vector field for the given level set
     */
    normal_vector_operation.solve(solution_levelset);

    VectorType rhs;

    scratch_data.initialize_dof_vector(rhs, curv_dof_idx);
    scratch_data.initialize_dof_vector(solution_curvature, curv_dof_idx);

    // no need to compute curvature in 1d
    if (dim == 1)
      return;

    int iter = 0;

    solution_levelset.update_ghost_values();
    normal_vector_operation.get_solution_normal_vector().update_ghost_values();

    if (curvature_data.linear_solver.do_matrix_free)
      {
        AssertThrow(preconditioner_matrixfree, ExcNotImplemented());

        curvature_operator->create_rhs(rhs, normal_vector_operation.get_solution_normal_vector());

        if (curvature_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          {
            iter = LinearSolver::solve<VectorType>(*curvature_operator,
                                                   solution_curvature,
                                                   rhs,
                                                   curvature_data.linear_solver,
                                                   *diag_preconditioner_matrixfree);
          }
        else
          {
            iter = LinearSolver::solve<VectorType>(*curvature_operator,
                                                   solution_curvature,
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
                                               solution_curvature,
                                               rhs,
                                               curvature_data.linear_solver);
      }

    solution_levelset.zero_out_ghost_values();
    normal_vector_operation.get_solution_normal_vector().zero_out_ghost_values();

    scratch_data.get_constraint(curv_dof_idx).distribute(solution_curvature);

    const unsigned int        verbosity_l2_norm = dim > 1 ? 0 : 1;
    const ConditionalOStream &pcout =
      scratch_data.get_pcout(std::max(curvature_data.verbosity_level, verbosity_l2_norm));


    Journal::print_formatted_norm(pcout,
                                  VectorTools::compute_L2_norm<dim>(
                                    solution_curvature, scratch_data, curv_dof_idx, curv_quad_idx),
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
    return solution_curvature;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  CurvatureOperation<dim>::get_curvature()
  {
    return solution_curvature;
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
    if (!curvature_data.linear_solver.do_matrix_free)
      curvature_operator->initialize_matrix_based(scratch_data);

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
