#include <meltpooldg/level_set/curvature_DG_operation.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>


namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  CurvatureDGOperation<dim, number>::CurvatureDGOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const unsigned int                   curvature_dof_idx_in,
    const unsigned int                   curvature_quad_idx_in,
    const BlockVectorType               &solution_normal_vector_in,
    const CurvatureData<number>         &curvature_data_in)
    : scratch_data(scratch_data_in)
    , solution_normal_vector(solution_normal_vector_in)
    , curvature_data(curvature_data_in)
    , solution_history(1)
    , curvature_dof_idx(curvature_dof_idx_in)
    , curvature_quad_idx(curvature_quad_idx_in)
    , helmholtz_operator(scratch_data_in,
                         curvature_dof_idx,
                         curvature_quad_idx,
                         curvature_data.filter_parameter,
                         curvature_data.curvature_DG_specific_data.penalty_factor,
                         curvature_data.linear_solver.preconditioner_type)
  {}

  template <int dim, typename number>
  void
  CurvatureDGOperation<dim, number>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, curvature_dof_idx); });

    helmholtz_operator.reinit();
  }


  template <int dim, typename number>
  void
  CurvatureDGOperation<dim, number>::solve()
  {
    const bool update_ghosts = !solution_normal_vector.has_ghost_elements();
    if (update_ghosts)
      solution_normal_vector.update_ghost_values();

    VectorType right_hand_side;
    scratch_data.initialize_dof_vector(right_hand_side, curvature_dof_idx);

    scratch_data.get_matrix_free().loop(
      &CurvatureDGOperation<dim, number>::right_hand_side_domain,
      &CurvatureDGOperation<dim, number>::right_hand_side_inner_face<0>,
      &CurvatureDGOperation<dim, number>::right_hand_side_boundary_face<0>,
      this,
      right_hand_side,
      solution_normal_vector,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::unspecified,
      MatrixFree<dim, number>::DataAccessOnFaces::unspecified);

    LinearSolver::solve<VectorType>(helmholtz_operator,
                                    solution_history.get_current_solution(),
                                    right_hand_side,
                                    curvature_data.linear_solver);
  }

  template <int dim, typename number>
  void
  CurvatureDGOperation<dim, number>::right_hand_side_domain(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const CurvatureDGOperation::BlockVectorType &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, dim, number> eval_normal(data, curvature_dof_idx, curvature_quad_idx);
    FECellIntegrator<dim, 1, number>   eval(data, curvature_dof_idx, curvature_quad_idx);


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_normal.reinit(cell);
        eval_normal.read_dof_values_plain(src);
        eval_normal.evaluate(EvaluationFlags::values);

        eval.reinit(cell);

        /**
         * 1) Normalize normal vector
         */
        for (unsigned int i = 0; i < eval_normal.dofs_per_component; ++i)
          {
            if constexpr (dim > 1)
              {
                const Tensor<1, dim, VectorizedArray<number>> n_phi =
                  MeltPoolDG::VectorTools::normalize<dim>(eval_normal.get_dof_value(i), 1.0e-16

                  );
                eval_normal.submit_dof_value(n_phi, i);
              }
            else
              {
                const VectorizedArray<number> n_phi =
                  compare_and_apply_mask<SIMDComparison::greater_than>(eval_normal.get_dof_value(i),
                                                                       1.0e-16,
                                                                       1.0,
                                                                       -1.0);

                eval_normal.submit_dof_value(n_phi, i);
              }
          }

        /**
         * 2) fill right hand side
         */
        eval_normal.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            auto flux = eval_normal.get_gradient(q)[0][0];
            for (unsigned int direction = 1; direction < dim; direction++)
              {
                flux += eval_normal.get_gradient(q)[direction][direction];
              }

            eval.submit_value(-flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  CurvatureDGOperation<dim, number>::get_curvature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  CurvatureDGOperation<dim, number>::get_curvature()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  void
  CurvatureDGOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template class CurvatureDGOperation<1, double>;
  template class CurvatureDGOperation<2, double>;
  template class CurvatureDGOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
