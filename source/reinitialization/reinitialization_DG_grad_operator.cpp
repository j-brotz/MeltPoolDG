#include <meltpooldg/reinitialization/reinitialization_DG_grad_operator.hpp>



namespace MeltPoolDG::LevelSet
{

  template <int dim, typename Number>
  RIGradOperator<dim, Number>::RIGradOperator(const MeltPoolDG::ScratchData<dim> &scratch_data_in,
                                              const unsigned int                  reinit_dof_idx_in,
                                              const unsigned int reinit_quad_idx_in)
    : scratch_data(scratch_data_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
  {}

  template <int dim, typename Number>
  template <bool is_right, uint component>
  void
  RIGradOperator<dim, Number>::apply(const VectorType &src, VectorType &dst)
  {
    scratch_data.get_matrix_free().loop(
      &RIGradOperator<dim, Number>::local_apply_domain<component>,
      &RIGradOperator<dim, Number>::local_apply_inner_face<is_right, component>,
      &RIGradOperator<dim, Number>::local_apply_boundary_face<is_right, component>,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, Number>::DataAccessOnFaces::values,
      MatrixFree<dim, Number>::DataAccessOnFaces::values);

    scratch_data.get_matrix_free().cell_loop(&RIGradOperator<dim>::local_apply_inverse_mass_matrix,
                                             this,
                                             dst,
                                             dst);
  }

  template <int dim, typename Number>
  void
  RIGradOperator<dim, Number>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, Number> eval(data, reinit_dof_idx, reinit_quad_idx);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, Number> inverse(eval);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);

        inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

        eval.set_dof_values(dst);
      }
  }



  template <int dim, typename Number>
  template <uint component>
  void
  RIGradOperator<dim, Number>::local_apply_domain(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, Number> eval(data, reinit_dof_idx, reinit_quad_idx);

    const Tensor<1, dim, Number> unit_vector = Point<dim>::unit_vector(component);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u    = eval.get_value(q);
            const auto flux = unit_vector * u;
            eval.submit_gradient(-flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename Number>
  template <bool is_right, uint component>
  void
  RIGradOperator<dim, Number>::local_apply_inner_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, Number> eval_minus(data, true, reinit_dof_idx, reinit_quad_idx);
    FEFaceIntegrator<dim, 1, Number> eval_plus(data, false, reinit_dof_idx, reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_plus.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto u_minus             = eval_minus.get_value(q);
            const auto u_plus              = eval_plus.get_value(q);
            const auto normal_vector_minus = eval_minus.get_normal_vector(q);

            const auto flux = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
              normal_vector_minus[component],
              0.,
              normal_vector_minus[component] * ((is_right == true) ? u_plus : u_minus),
              normal_vector_minus[component] * ((is_right == true) ? u_minus : u_plus));

            eval_minus.submit_value(flux, q);
            eval_plus.submit_value(-flux, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }


  template <int dim, typename Number>
  template <bool is_right, uint component>
  void
  RIGradOperator<dim, Number>::local_apply_boundary_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, Number> eval_minus(data, true, reinit_dof_idx, reinit_quad_idx);


    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto u_minus             = eval_minus.get_value(q);
            const auto normal_vector_minus = eval_minus.get_normal_vector(q);

            // Compute the outer solution value
            const auto u_plus = u_minus;

            // Compute the flux
            const auto flux = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
              normal_vector_minus[component],
              0.,
              normal_vector_minus[component] * ((is_right == true) ? u_plus : u_minus),
              normal_vector_minus[component] * ((is_right == true) ? u_minus : u_plus));

            eval_minus.submit_value(flux, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template class RIGradOperator<1>;
  template class RIGradOperator<2>;
  template class RIGradOperator<3>;

  //@todo: Is there a more elegant way??
  template void
  RIGradOperator<1, double>::apply<true, 0>(const VectorType &, VectorType &);
  template void
  RIGradOperator<1, double>::apply<false, 0>(const VectorType &, VectorType &);

  template void
  RIGradOperator<2, double>::apply<true, 1>(const VectorType &, VectorType &);
  template void
  RIGradOperator<2, double>::apply<false, 1>(const VectorType &, VectorType &);

  template void
  RIGradOperator<2, double>::apply<true, 0>(const VectorType &, VectorType &);
  template void
  RIGradOperator<2, double>::apply<false, 0>(const VectorType &, VectorType &);

  template void
  RIGradOperator<3, double>::apply<true, 2>(const VectorType &, VectorType &);
  template void
  RIGradOperator<3, double>::apply<false, 2>(const VectorType &, VectorType &);

  template void
  RIGradOperator<3, double>::apply<true, 1>(const VectorType &, VectorType &);
  template void
  RIGradOperator<3, double>::apply<false, 1>(const VectorType &, VectorType &);

  template void
  RIGradOperator<3, double>::apply<true, 0>(const VectorType &, VectorType &);
  template void
  RIGradOperator<3, double>::apply<false, 0>(const VectorType &, VectorType &);
} // namespace MeltPoolDG::LevelSet
