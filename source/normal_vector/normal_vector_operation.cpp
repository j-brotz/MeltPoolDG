#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::NormalVector
{
  template <int dim>
  NormalVectorOperation<dim>::NormalVectorOperation(
    const ScratchData<dim> &        scratch_data_in,
    const NormalVectorData<double> &normal_vector_data,
    const VectorType &              solution_level_set,
    const unsigned int              normal_dof_idx_in,
    const unsigned int              normal_quad_idx_in,
    const unsigned int              ls_dof_idx_in)
    : scratch_data(scratch_data_in)
    , normal_vector_data(normal_vector_data)
    , solution_level_set(solution_level_set)
    , normal_dof_idx(normal_dof_idx_in)
    , normal_quad_idx(normal_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
  {
    AssertThrow(normal_vector_data.linear_solver.solver_type == LinearSolverType::CG,
                ExcMessage("The normal vector operation only supports the CG solver type."));

    if (!normal_vector_operator)
      create_operator();

    reinit();
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::reinit()
  {
    scratch_data.initialize_dof_vector(solution_normal_vector, normal_dof_idx);
    scratch_data.initialize_dof_vector(solution_normal_vector_old, normal_dof_idx);
    if (normal_vector_data.predictor.type == PredictorType::linear_extrapolation)
      {
        scratch_data.initialize_dof_vector(solution_normal_vector_old, normal_dof_idx);
        scratch_data.initialize_dof_vector(solution_normal_vector_predictor, normal_dof_idx);
      }
    scratch_data.initialize_dof_vector(rhs, normal_dof_idx);

    if (!normal_vector_data.linear_solver.do_matrix_free)
      normal_vector_operator->initialize_matrix_based(scratch_data);

    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        /*
         * precompute preconditioner
         */
        if (normal_vector_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          diag_preconditioner_matrixfree =
            preconditioner_matrixfree->compute_block_diagonal_preconditioner();
      }
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::solve()
  {
    if (normal_vector_data.predictor.type == PredictorType::linear_extrapolation)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            UtilityFunctions::compute_linear_predictor(solution_normal_vector.block(d),
                                                       solution_normal_vector_old.block(d),
                                                       solution_normal_vector_predictor.block(d),
                                                       1 /*not time-dependent*/,
                                                       1 /*not time-dependent*/);
          }

        solution_normal_vector_old.swap(solution_normal_vector);
        solution_normal_vector.swap(solution_normal_vector_predictor);

        // apply hanging node constraints to predictor
        for (unsigned int d = 0; d < dim; ++d)
          scratch_data.get_constraint(normal_dof_idx).distribute(solution_normal_vector.block(d));
      }

    int iter = 0;

    solution_level_set.update_ghost_values();

    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        AssertThrow(preconditioner_matrixfree, ExcNotImplemented());

        normal_vector_operator->create_rhs(rhs, solution_level_set);

        if (diag_preconditioner_matrixfree)
          iter = LinearSolver::solve<BlockVectorType>(*normal_vector_operator,
                                                      solution_normal_vector,
                                                      rhs,
                                                      normal_vector_data.linear_solver,
                                                      *diag_preconditioner_matrixfree);
        else if (normal_vector_data.linear_solver.preconditioner_type ==
                 PreconditionerType::Identity)
          iter = LinearSolver::solve<BlockVectorType>(*normal_vector_operator,
                                                      solution_normal_vector,
                                                      rhs,
                                                      normal_vector_data.linear_solver);
        else
          AssertThrow(false, ExcNotImplemented());
      }
    else
      {
        AssertThrow(
          !normal_vector_data.do_narrow_band,
          ExcMessage(
            "The computation of the normal vector in a narrow band is only implemented matrix-free."));

        normal_vector_operator->assemble_matrixbased(solution_level_set,
                                                     normal_vector_operator->get_system_matrix(),
                                                     rhs);

        for (unsigned int d = 0; d < dim; ++d)
          iter = LinearSolver::solve<VectorType>(normal_vector_operator->get_system_matrix(),
                                                 solution_normal_vector.block(d),
                                                 rhs.block(d),
                                                 normal_vector_data.linear_solver);
      }

    solution_level_set.zero_out_ghost_values();

    for (unsigned int d = 0; d < dim; ++d)
      scratch_data.get_constraint(normal_dof_idx).distribute(solution_normal_vector.block(d));

    const unsigned int        verbosity_l2_norm = dim > 1 ? 0 : 1;
    const ConditionalOStream &pcout =
      scratch_data.get_pcout(std::max(normal_vector_data.verbosity_level, verbosity_l2_norm));

    for (unsigned int d = 0; d < dim; ++d)
      Journal::print_formatted_norm(
        pcout,
        MeltPoolDG::VectorTools::compute_L2_norm<dim>(
          solution_normal_vector.block(d), scratch_data, normal_dof_idx, normal_quad_idx),
        "normal_" + std::to_string(d),
        "normal_vector",
        11 /*precision*/
      );

    Journal::print_line(scratch_data.get_pcout(1),
                        "     * CG: i = " + std::to_string(iter),
                        "normal_vector");
  }

  template <int dim>
  const typename NormalVectorOperation<dim>::BlockVectorType &
  NormalVectorOperation<dim>::get_solution_normal_vector() const
  {
    return solution_normal_vector;
  }

  template <int dim>
  typename NormalVectorOperation<dim>::BlockVectorType &
  NormalVectorOperation<dim>::get_solution_normal_vector()
  {
    return solution_normal_vector;
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    for (unsigned int d = 0; d < dim; ++d)
      vectors.push_back(&solution_normal_vector.block(d));

    if (normal_vector_data.predictor.type == PredictorType::linear_extrapolation)
      {
        for (unsigned int d = 0; d < dim; ++d)
          vectors.push_back(&solution_normal_vector_old.block(d));
      }
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::create_operator()
  {
    normal_vector_operator =
      std::make_unique<NormalVectorOperator<dim>>(scratch_data,
                                                  normal_vector_data,
                                                  normal_dof_idx,
                                                  normal_quad_idx,
                                                  ls_dof_idx,
                                                  normal_vector_data.do_narrow_band,
                                                  &solution_level_set);
    /*
     *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
     *  apply it to the system matrix. This functionality is part of the OperatorBase class.
     */
    if (!normal_vector_data.linear_solver.do_matrix_free)
      normal_vector_operator->initialize_matrix_based(scratch_data);
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
        /*
         * precompute system matrix
         */
        if (normal_vector_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          diag_preconditioner_matrixfree =
            preconditioner_matrixfree->compute_block_diagonal_preconditioner();
      }
  }


  template class NormalVectorOperation<1>;
  template class NormalVectorOperation<2>;
  template class NormalVectorOperation<3>;
} // namespace MeltPoolDG::NormalVector
