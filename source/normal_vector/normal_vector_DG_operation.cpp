#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/normal_vector/normal_vector_DG_operation.hpp>


namespace MeltPoolDG::LevelSet
{
  template <int dim, typename Number>
  NormalVectorDGOperation<dim, Number>::NormalVectorDGOperation(
    const ScratchData<dim>         &scratch_data_in,
    const unsigned int              normal_dof_idx_in,
    const unsigned int              normal_quad_idx_in,
    const VectorType               &solution_level_set_in,
    const NormalVectorData<Number> &normal_vector_data_in)
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

  template <int dim, typename Number>
  void
  NormalVectorDGOperation<dim, Number>::reinit()
  {
    solution_history.apply(
      [this](BlockVectorType &v) { scratch_data.initialize_dof_vector(v, normal_dof_idx); });

    helmholtz_operator.reinit();
  }


  template <int dim, typename Number>

  void
  NormalVectorDGOperation<dim, Number>::solve()
  {
    VectorType right_hand_side;
    scratch_data.initialize_dof_vector(right_hand_side, normal_dof_idx);

    /**
    * The right hand side is filled for each dimension individually.
    * Afterwards the system is is solved for each component of the normal vector
    */

    if constexpr (dim > 0)
      {
       scratch_data.get_matrix_free().loop(
        &NormalVectorDGOperation<dim, Number>::right_hand_side_domain<0>,
        &NormalVectorDGOperation<dim, Number>::right_hand_side_inner_face<0>,
        &NormalVectorDGOperation<dim, Number>::right_hand_side_boundary_face<0>,
        this,
        right_hand_side,
        solution_level_set,
        true,
        MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
        MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);

       LinearSolver::solve<VectorType>(helmholtz_operator,
                                       solution_history.get_current_solution().block(0),
                                       right_hand_side,
                                       normal_vector_data.linear_solver,
                                       *helmholtz_operator.get_preconditioner());
      }

    if constexpr (dim > 1)
      {
        scratch_data.get_matrix_free().loop(
          &NormalVectorDGOperation<dim, Number>::right_hand_side_domain<1>,
          &NormalVectorDGOperation<dim, Number>::right_hand_side_inner_face<1>,
          &NormalVectorDGOperation<dim, Number>::right_hand_side_boundary_face<1>,
          this,
          right_hand_side,
          solution_level_set,
          true,
          MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
          MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);

        LinearSolver::solve<VectorType>(helmholtz_operator,
                                        solution_history.get_current_solution().block(1),
                                        right_hand_side,
                                        normal_vector_data.linear_solver,
                                        *helmholtz_operator.get_preconditioner());
      }

    if constexpr (dim > 2)
      {
        scratch_data.get_matrix_free().loop(
          &NormalVectorDGOperation<dim, Number>::right_hand_side_domain<2>,
          &NormalVectorDGOperation<dim, Number>::right_hand_side_inner_face<2>,
          &NormalVectorDGOperation<dim, Number>::right_hand_side_boundary_face<2>,
          this,
          right_hand_side,
          solution_level_set,
          true,
          MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
          MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);

        LinearSolver::solve<VectorType>(helmholtz_operator,
                                        solution_history.get_current_solution().block(2),
                                        right_hand_side,
                                        normal_vector_data.linear_solver);
      }

      if constexpr (dim>3)
      {
        DEAL_II_NOT_IMPLEMENTED();
      }
  }

  template <int dim, typename Number>
  template <uint direction>
  void
  NormalVectorDGOperation<dim, Number>::right_hand_side_domain(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, Number> eval(data, normal_dof_idx, normal_quad_idx);

    const Tensor<1, dim, Number> unit_vector = Point<dim>::unit_vector(direction);


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u    = eval.get_value(q);
            const auto flux = u * unit_vector;

            eval.submit_gradient(-flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename Number>
  template <uint direction>
  void
  NormalVectorDGOperation<dim, Number>::right_hand_side_inner_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, Number> eval_minus(data, true, normal_dof_idx, normal_quad_idx);
    FEFaceIntegrator<dim, 1, Number> eval_plus(data, false, normal_dof_idx, normal_quad_idx);

    const Tensor<1, dim, Number> unit_vector = Point<dim>::unit_vector(direction);


    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_plus.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = eval_plus.get_value(q);

            const auto average = 0.5 * (u_minus + u_plus) * unit_vector;

            eval_minus.submit_value(eval_minus.get_normal_vector(q) * average, q);
            eval_plus.submit_value(-eval_minus.get_normal_vector(q) * average, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename Number>
  template <uint direction>
  void
  NormalVectorDGOperation<dim, Number>::right_hand_side_boundary_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, Number> eval_minus(data, true, normal_dof_idx, normal_quad_idx);

    const Tensor<1, dim, Number> unit_vector = Point<dim>::unit_vector(direction);


    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = u_minus;

            const auto average = 0.5 * (u_minus + u_plus) * unit_vector;

            eval_minus.submit_value(eval_minus.get_normal_vector(q) * average, q);
          }
        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename Number>

  const typename NormalVectorDGOperation<dim, Number>::BlockVectorType &
  NormalVectorDGOperation<dim, Number>::get_solution_normal_vector() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename Number>

  typename NormalVectorDGOperation<dim, Number>::BlockVectorType &
  NormalVectorDGOperation<dim, Number>::get_solution_normal_vector()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename Number>

  void
  NormalVectorDGOperation<dim, Number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<Number> *> &vectors)
  {
    solution_history.apply([&](BlockVectorType &v) {
      for (unsigned int d = 0; d < dim; ++d)
        vectors.push_back(&v.block(d));
    });
  }

  template class NormalVectorDGOperation<1>;
  template class NormalVectorDGOperation<2>;
  template class NormalVectorDGOperation<3>;
} // namespace MeltPoolDG::LevelSet
