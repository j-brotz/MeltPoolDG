#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>

#include <meltpooldg/utilities/amr_indicators.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

/**
 * Helper function that returns the values at a given quadrature point (specified by `q_index`)
 * using the provided finite element evaluation object (`fe_eval`) and ensures that the result
 * is always returned as a `dealii::Tensor`.
 *
 * In deal.II, `FEValues` and similar objects return a `VectorizedArray` directly when there is
 * only a single component. However, the algorithms in this codebase are written to operate on
 * tensors for both 2D and 3D problems and for systems with multiple components. This function
 * provides a uniform interface by converting single-component vectorized values into a
 * `Tensor<1,1,VectorizedArray>` while leaving multi-component values as-is.
 *
 * @param fe_eval The finite element evaluation object.
 * @param q_index Index of the quadrature point.

 * @return Value at the specified quadrature point as a `dealii::Tensor`.
 */
template <typename FeEval>
dealii::Tensor<1, FeEval::n_components, dealii::VectorizedArray<typename FeEval::number_type>>
fe_evaluation_tensor_value_at_q(const FeEval &fe_eval, const unsigned q_index)
{
  using ValueType           = typename FeEval::value_type;
  using VectorizedArrayType = dealii::VectorizedArray<typename FeEval::number_type>;

  if constexpr (std::is_same_v<ValueType, VectorizedArrayType>)
    {
      dealii::Tensor<1, 1, VectorizedArrayType> t;
      t[0] = fe_eval.get_value(q_index);
      return t;
    }
  else
    return fe_eval.get_value(q_index);
}

template <int dim, typename number, template <typename> class BinaryOp>
void
MeltPoolDG::AMR::BinaryOpIndicatorComposite<dim, number, BinaryOp>::add_indicator(
  std::unique_ptr<AMRIndicatorBase<dim, number>> &&indicator,
  number                                           weight)
{
  indicators_and_weights.emplace_back(std::move(indicator), weight);
}

template <int dim, typename number, template <typename> class BinaryOp>
auto
MeltPoolDG::AMR::BinaryOpIndicatorComposite<dim, number, BinaryOp>::compute_indicator(
  const dealii::Triangulation<dim> &tria) -> VectorType
{
  AssertThrow(not indicators_and_weights.empty(),
              dealii::ExcMessage("No AMR indicator has been added to the indicator composite."));

  auto check_partitioning = [&tria](const VectorType &indicator) {
    Assert(
      tria.n_active_cells() == indicator.size(),
      dealii::ExcMessage(
        "The indicator has returned a vector which does not has the same number of elements as the triangulation has local active cells."));
  };

  VectorType indicator = indicators_and_weights[0].indicator->compute_indicator(tria);
  indicator *= indicators_and_weights[0].weight;

  check_partitioning(indicator);
  for (unsigned i = 1; i < this->indicators_and_weights.size(); ++i)
    {
      VectorType temp_indicator = indicators_and_weights[i].indicator->compute_indicator(tria);
      check_partitioning(temp_indicator);
      for (unsigned j = 0; j < indicator.size(); ++j)
        indicator[j] =
          BinaryOp<number>()(indicator[j], indicators_and_weights[i].weight * temp_indicator[j]);
    }

  return indicator;
}

template <int dim, typename number, template <typename> class BinaryOp>
bool
MeltPoolDG::AMR::BinaryOpIndicatorComposite<dim, number, BinaryOp>::empty() const
{
  return indicators_and_weights.empty();
}

template <int dim, typename number, int n_dof_components, int n_indicator_components>
MeltPoolDG::AMR::SSEDIndicator<dim, number, n_dof_components, n_indicator_components>::
  SSEDIndicator(const MatrixFreeContext<dim, number> matrix_free_context,
                const VectorType                    &solution,
                const ExtractIndicatorDofValuesFn    extract_indicator_dof_values,
                const unsigned                       fe_index,
                const unsigned                       fe_degree)
  : matrix_free_context(matrix_free_context)
  , solution(solution)
  , extract_indicator_dof_values(extract_indicator_dof_values)
  , fe_index(fe_index)
{
  AssertThrow(fe_degree > 1,
              dealii::ExcMessage("The SSED indicator requires an fe degree greater than one."));

  dealii::FullMatrix<number> transformation_matrix(fe_degree + 1);
  dealii::FETools::get_projection_matrix(dealii::FE_DGQ<1>(fe_degree),
                                         dealii::FE_DGQLegendre<1>(fe_degree),
                                         transformation_matrix);
  linearized_matrix.resize((fe_degree + 1) * (fe_degree + 1));
  for (unsigned i = 0; i < fe_degree + 1; ++i)
    for (unsigned j = 0; j < fe_degree + 1; ++j)
      linearized_matrix[j + ((fe_degree + 1) * i)] = transformation_matrix[j][i];
}

template <int dim, typename number, int n_dof_components, int n_indicator_components>
dealii::Vector<number>
MeltPoolDG::AMR::SSEDIndicator<dim, number, n_dof_components, n_indicator_components>::
  compute_indicator(const dealii::Triangulation<dim> &tria)
{
  const unsigned fe_degree =
    matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx).get_fe(fe_index).degree;

  dealii::AlignedVector<VectorizedArrayType> indicator;
  matrix_free_context.mf.initialize_cell_data_vector(indicator);

  if (fe_degree > 1)
    {
      std::function<void(const dealii::MatrixFree<dim, number>      &mf,
                         dealii::AlignedVector<VectorizedArrayType> &indicator,
                         const VectorType                           &solution,
                         const std::pair<unsigned, unsigned>        &cell_range)>
        cell_op = [&](const dealii::MatrixFree<dim, number>      &mf,
                      dealii::AlignedVector<VectorizedArrayType> &indicator,
                      const VectorType                           &solution,
                      const std::pair<unsigned, unsigned>        &cell_range) {
          local_apply_cell(mf, indicator, solution, cell_range);
        };

      matrix_free_context.mf.cell_loop(cell_op, indicator, solution);
    }

  return VectorTools::convert_matrix_free_cell_aligned_vector_to_vector<dim, number, dim + 2>(
    matrix_free_context, indicator, tria);
}

template <int dim, typename number, int n_dof_components, int n_indicator_components>
void
MeltPoolDG::AMR::SSEDIndicator<dim, number, n_dof_components, n_indicator_components>::
  local_apply_cell(const dealii::MatrixFree<dim, number> &,
                   dealii::AlignedVector<VectorizedArrayType> &dst,
                   const VectorType                           &src,
                   const std::pair<unsigned, unsigned>        &cell_range)
{
  FECellIntegrator<dim, n_dof_components, number> phi_dgq(matrix_free_context.mf,
                                                          matrix_free_context.dof_idx,
                                                          matrix_free_context.quad_idx);
  FECellIntegrator<dim, n_dof_components, number> phi_leg(matrix_free_context.mf,
                                                          matrix_free_context.dof_idx,
                                                          matrix_free_context.quad_idx);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      VectorizedArrayType ssed_error = 0.;
      phi_dgq.reinit(cell);
      phi_leg.reinit(cell);

      phi_dgq.read_dof_values(src);

      VectorizedArrayType cell_volume = 0.;
      const unsigned fe_degree = matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx)
                                   .get_fe(phi_dgq.get_active_fe_index())
                                   .degree;
      // transform cell solution values from modal basis to modal basis
      dealii::internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_general,
                                                    dealii::internal::EvaluatorQuantity::value,
                                                    dim,
                                                    0,
                                                    0>::template do_forward<VectorizedArrayType,
                                                                            VectorizedArrayType>(
        n_dof_components,
        linearized_matrix,
        phi_dgq.begin_dof_values(),
        phi_leg.begin_dof_values(),
        fe_degree + 1,
        fe_degree + 1);

      // only keep highest modal basis
      for (unsigned i = 0; i < std::pow(fe_degree, dim); ++i)
        phi_leg.submit_dof_value(dealii::Tensor<1, n_dof_components, VectorizedArrayType>(), i);

      phi_leg.evaluate(dealii::EvaluationFlags::values);

      // integrate over cell
      for (const unsigned int q : phi_dgq.quadrature_point_indices())
        {
          const dealii::Tensor<1, n_indicator_components, VectorizedArrayType> u_q =
            extract_indicator_dof_values(fe_evaluation_tensor_value_at_q(phi_leg, q));
          ssed_error += phi_dgq.JxW(q) * u_q * u_q;
          cell_volume += phi_dgq.JxW(q);
        }

      ssed_error /= cell_volume;
      phi_dgq.set_cell_data(dst, std::sqrt(ssed_error));
    }
}

template <int dim, typename number, int n_dof_components, int n_indicator_components>
MeltPoolDG::AMR::JumpIndicator<dim, number, n_dof_components, n_indicator_components>::
  JumpIndicator(const MatrixFreeContext<dim, number> matrix_free_context,
                const VectorType                    &solution,
                const ExtractIndicatorDofValuesFn    extract_indicator_dof_values,
                const unsigned                       fe_index)
  : matrix_free_context(matrix_free_context)
  , solution(solution)
  , extract_indicator_dof_values(extract_indicator_dof_values)
  , fe_index(fe_index)
{
  // Jump indicator only works for DG elements.
  for (unsigned int sub_fe_idx = 0;
       sub_fe_idx < matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx)
                      .get_fe(fe_index)
                      .n_components();
       ++sub_fe_idx)
    AssertThrow(dynamic_cast<const dealii::FE_DGQ<dim> *>(
                  &matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx)
                     .get_fe(fe_index)
                     .get_sub_fe(sub_fe_idx, 1)),
                dealii::ExcMessage("The jump indicator only works with DG elements."));
}

template <int dim, typename number, int n_dof_components, int n_indicator_components>
dealii::Vector<number>
MeltPoolDG::AMR::JumpIndicator<dim, number, n_dof_components, n_indicator_components>::
  compute_indicator(const dealii::Triangulation<dim> &tria)
{
  using LocalOpType = std::function<void(const dealii::MatrixFree<dim, number>      &mf,
                                         dealii::AlignedVector<VectorizedArrayType> &indicator,
                                         const VectorType                           &solution,
                                         const std::pair<unsigned, unsigned>        &cell_range)>;

  dealii::AlignedVector<VectorizedArrayType> indicator;
  matrix_free_context.mf.initialize_cell_data_vector(indicator);

  LocalOpType cell_op = [](const dealii::MatrixFree<dim, number> &,
                           dealii::AlignedVector<VectorizedArrayType> &,
                           const VectorType &,
                           const std::pair<unsigned, unsigned> &) {};

  LocalOpType face_op = std::bind_front(
    &JumpIndicator<dim, number, n_dof_components, n_indicator_components>::local_apply_face, this);

  LocalOpType boundary_face_op = [](const dealii::MatrixFree<dim, number> &,
                                    dealii::AlignedVector<VectorizedArrayType> &,
                                    const VectorType &,
                                    const std::pair<unsigned, unsigned> &) {};

  matrix_free_context.mf.loop(cell_op, face_op, boundary_face_op, indicator, solution);

  return VectorTools::
    convert_matrix_free_cell_aligned_vector_to_vector<dim, number, n_dof_components>(
      matrix_free_context, indicator, tria);
}

template <int dim, typename number, int n_dof_components, int n_indicator_components>
void
MeltPoolDG::AMR::JumpIndicator<dim, number, n_dof_components, n_indicator_components>::
  local_apply_face(const dealii::MatrixFree<dim, number> &,
                   dealii::AlignedVector<VectorizedArrayType> &dst,
                   const VectorType                           &src,
                   const std::pair<unsigned, unsigned>        &face_range) const
{
  FEFaceIntegrator<dim, n_dof_components, number> phi_m(matrix_free_context.mf,
                                                        true,
                                                        matrix_free_context.dof_idx,
                                                        matrix_free_context.quad_idx);
  FEFaceIntegrator<dim, n_dof_components, number> phi_p(matrix_free_context.mf,
                                                        false,
                                                        matrix_free_context.dof_idx,
                                                        matrix_free_context.quad_idx);

  for (unsigned face = face_range.first; face < face_range.second; ++face)
    {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, dealii::EvaluationFlags::values);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, dealii::EvaluationFlags::values);

      VectorizedArrayType face_area       = 0.;
      VectorizedArrayType face_jump_error = 0.;
      for (const unsigned q : phi_m.quadrature_point_indices())
        {
          const auto w_p = extract_indicator_dof_values(fe_evaluation_tensor_value_at_q(phi_p, q));
          const auto w_m = extract_indicator_dof_values(fe_evaluation_tensor_value_at_q(phi_m, q));

          face_jump_error += phi_m.JxW(q) * (w_m - w_p) * (w_m - w_p);

          face_area += phi_m.JxW(q);
        }
      constexpr int       n_faces            = 2 * dim;
      VectorizedArrayType current_cell_error = phi_m.read_cell_data(dst);
      current_cell_error += 0.25 / n_faces * std::sqrt(face_jump_error / face_area);
      phi_m.set_cell_data(dst, current_cell_error);
    }
}