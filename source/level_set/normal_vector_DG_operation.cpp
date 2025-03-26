#include <meltpooldg/level_set/normal_vector_DG_operation.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>


namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  NormalVectorDGOperation<dim, number>::NormalVectorDGOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const unsigned int                   normal_dof_idx_in,
    const unsigned int                   normal_quad_idx_in,
    const VectorType                    &solution_level_set_in,
    const NormalVectorData<number>      &normal_vector_data_in)
    : scratch_data(scratch_data_in)
    , solution_level_set(solution_level_set_in)
    , normal_vector_data(normal_vector_data_in)
    , solution_history(1)
    , normal_dof_idx(normal_dof_idx_in)
    , normal_quad_idx(normal_quad_idx_in)
    , helmholtz_operator(scratch_data_in,
                         normal_dof_idx,
                         normal_quad_idx,
                         normal_vector_data.filter_parameter,
                         normal_vector_data.normal_DG_specific_data.penalty_factor,
                         normal_vector_data.linear_solver.preconditioner_type)
  {}

  template <int dim, typename number>
  void
  NormalVectorDGOperation<dim, number>::reinit()
  {
    solution_history.apply(
      [this](BlockVectorType &v) { scratch_data.initialize_dof_vector(v, normal_dof_idx); });

    helmholtz_operator.reinit();
  }


  template <int dim, typename number>

  void
  NormalVectorDGOperation<dim, number>::solve()
  {
    VectorType right_hand_side;
    scratch_data.initialize_dof_vector(right_hand_side, normal_dof_idx);

    /**
     * The right hand side is filled for each dimension individually.
     * Afterwards the system is is solved for each component of the normal vector
     */

    if constexpr (dim > 0)
      {
        scratch_data.get_matrix_free().cell_loop(
          &NormalVectorDGOperation<dim, number>::right_hand_side_domain<0>,
          this,
          right_hand_side,
          solution_level_set,
          true);

        LinearSolver::solve<VectorType>(helmholtz_operator,
                                        solution_history.get_current_solution().block(0),
                                        right_hand_side,
                                        normal_vector_data.linear_solver,
                                        helmholtz_operator.get_preconditioner());
      }

    if constexpr (dim > 1)
      {
        scratch_data.get_matrix_free().cell_loop(
          &NormalVectorDGOperation<dim, number>::right_hand_side_domain<1>,
          this,
          right_hand_side,
          solution_level_set,
          true);

        LinearSolver::solve<VectorType>(helmholtz_operator,
                                        solution_history.get_current_solution().block(1),
                                        right_hand_side,
                                        normal_vector_data.linear_solver,
                                        helmholtz_operator.get_preconditioner());
      }

    if constexpr (dim > 2)
      {
        scratch_data.get_matrix_free().cell_loop(
          &NormalVectorDGOperation<dim, number>::right_hand_side_domain<2>,
          this,
          right_hand_side,
          solution_level_set,
          true);

        LinearSolver::solve<VectorType>(helmholtz_operator,
                                        solution_history.get_current_solution().block(2),
                                        right_hand_side,
                                        normal_vector_data.linear_solver);
      }

    if constexpr (dim > 3)
      {
        DEAL_II_NOT_IMPLEMENTED();
      }
  }

  template <int dim, typename number>
  template <uint direction>
  void
  NormalVectorDGOperation<dim, number>::right_hand_side_domain(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, number> eval(data, normal_dof_idx, normal_quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            eval.submit_value(eval.get_gradient(q)[direction], q);
          }
        eval.integrate_scatter(EvaluationFlags::values, dst);
      }
  }


  template <int dim, typename number>

  const typename NormalVectorDGOperation<dim, number>::BlockVectorType &
  NormalVectorDGOperation<dim, number>::get_solution_normal_vector() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>

  typename NormalVectorDGOperation<dim, number>::BlockVectorType &
  NormalVectorDGOperation<dim, number>::get_solution_normal_vector()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>

  void
  NormalVectorDGOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    solution_history.apply([&](BlockVectorType &v) {
      for (unsigned int d = 0; d < dim; ++d)
        vectors.push_back(&v.block(d));
    });
  }

  template class NormalVectorDGOperation<1, double>;
  template class NormalVectorDGOperation<2, double>;
  template class NormalVectorDGOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
