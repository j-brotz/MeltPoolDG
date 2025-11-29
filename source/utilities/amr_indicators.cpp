#include <deal.II/base/exceptions.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_common.h>

#include <meltpooldg/utilities/amr_indicators.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <functional>
#include <utility>

template <int dim, typename number>
MeltPoolDG::AMR::SSEDIndicator<dim, number>::SSEDIndicator(
  const MatrixFreeContext<dim, number>                      matrix_free_context,
  const dealii::LinearAlgebra::distributed::Vector<number> &solution,
  const unsigned                                            fe_index,
  const unsigned                                            fe_degree)
  : matrix_free_context(matrix_free_context)
  , solution(solution)
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

template <int dim, typename number>
dealii::Vector<number>
MeltPoolDG::AMR::SSEDIndicator<dim, number>::compute_indicator(
  const dealii::Triangulation<dim> &tria)
{
  const unsigned fe_degree =
    matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx).get_fe(fe_index).degree;

  dealii::AlignedVector<dealii::VectorizedArray<number>> indicator;
  matrix_free_context.mf.initialize_cell_data_vector(indicator);

  if (fe_degree > 1)
    {
      std::function<void(const dealii::MatrixFree<dim, number>                    &mf,
                         dealii::AlignedVector<dealii::VectorizedArray<number>>   &indicator,
                         const dealii::LinearAlgebra::distributed::Vector<number> &solution,
                         const std::pair<unsigned, unsigned>                      &cell_range)>
        cell_op = [&](const dealii::MatrixFree<dim, number>                    &mf,
                      dealii::AlignedVector<dealii::VectorizedArray<number>>   &indicator,
                      const dealii::LinearAlgebra::distributed::Vector<number> &solution,
                      const std::pair<unsigned, unsigned>                      &cell_range) {
          local_apply_cell(mf, indicator, solution, cell_range);
        };

      matrix_free_context.mf.cell_loop(cell_op, indicator, solution);
    }

  return VectorTools::convert_matrix_free_cell_aligned_vector_to_vector<dim, number, dim + 2>(
    matrix_free_context, indicator, tria);
}

template <int dim, typename number>
void
MeltPoolDG::AMR::SSEDIndicator<dim, number>::local_apply_cell(
  const dealii::MatrixFree<dim, number> &,
  dealii::AlignedVector<dealii::VectorizedArray<number>>   &dst,
  const dealii::LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned, unsigned>                      &cell_range)
{
  FECellIntegrator<dim, dim + 2, number> phi_dgq(matrix_free_context.mf,
                                                 matrix_free_context.dof_idx,
                                                 matrix_free_context.quad_idx);
  FECellIntegrator<dim, dim + 2, number> phi_leg(matrix_free_context.mf,
                                                 matrix_free_context.dof_idx,
                                                 matrix_free_context.quad_idx);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      dealii::VectorizedArray<number> ssed_error = 0.;
      phi_dgq.reinit(cell);
      phi_leg.reinit(cell);

      phi_dgq.read_dof_values(src);

      dealii::VectorizedArray<number> cell_volume = 0.;
      const unsigned fe_degree = matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx)
                                   .get_fe(phi_dgq.get_active_fe_index())
                                   .degree;
      // transform cell solution values from modal basis to modal basis
      dealii::internal::FEEvaluationImplBasisChange<
        dealii::internal::evaluate_general,
        dealii::internal::EvaluatorQuantity::value,
        dim,
        0,
        0>::template do_forward<dealii::VectorizedArray<number>,
                                dealii::VectorizedArray<number>>(dim + 2,
                                                                 linearized_matrix,
                                                                 phi_dgq.begin_dof_values(),
                                                                 phi_leg.begin_dof_values(),
                                                                 fe_degree + 1,
                                                                 fe_degree + 1);

      // only keep highest modal basis
      for (unsigned i = 0; i < std::pow(fe_degree, dim); ++i)
        phi_leg.submit_dof_value(dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>(), i);

      phi_leg.evaluate(dealii::EvaluationFlags::values);

      // integrate over cell
      for (const unsigned int q : phi_dgq.quadrature_point_indices())
        {
          const auto                      u_q = phi_leg.get_value(q);
          dealii::VectorizedArray<double> intermediate(0.);
          for (int i = 1; i < dim + 1; ++i)
            intermediate += u_q[i] * u_q[i];
          ssed_error += intermediate * phi_dgq.JxW(q);
          cell_volume += phi_dgq.JxW(q);
        }

      ssed_error /= cell_volume;
      phi_dgq.set_cell_data(dst, std::sqrt(ssed_error));
    }
}

template <int dim, typename number>
MeltPoolDG::AMR::JumpIndicator<dim, number>::JumpIndicator(
  const MatrixFreeContext<dim, number>                      matrix_free_context,
  const dealii::LinearAlgebra::distributed::Vector<number> &solution,
  const unsigned                                            fe_index)
  : matrix_free_context(matrix_free_context)
  , solution(solution)
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

template <int dim, typename number>
dealii::Vector<number>
MeltPoolDG::AMR::JumpIndicator<dim, number>::compute_indicator(
  const dealii::Triangulation<dim> &tria)
{
  using LocalOpType =
    std::function<void(const dealii::MatrixFree<dim, number>                    &mf,
                       dealii::AlignedVector<dealii::VectorizedArray<number>>   &indicator,
                       const dealii::LinearAlgebra::distributed::Vector<number> &solution,
                       const std::pair<unsigned, unsigned>                      &cell_range)>;

  dealii::AlignedVector<dealii::VectorizedArray<number>> indicator;
  matrix_free_context.mf.initialize_cell_data_vector(indicator);

  LocalOpType cell_op = [](const dealii::MatrixFree<dim, number> &,
                           dealii::AlignedVector<dealii::VectorizedArray<number>> &,
                           const dealii::LinearAlgebra::distributed::Vector<number> &,
                           const std::pair<unsigned, unsigned> &) {};

  LocalOpType face_op = std::bind_front(&JumpIndicator<dim, number>::local_apply_face, this);

  LocalOpType boundary_face_op = [](const dealii::MatrixFree<dim, number> &,
                                    dealii::AlignedVector<dealii::VectorizedArray<number>> &,
                                    const dealii::LinearAlgebra::distributed::Vector<number> &,
                                    const std::pair<unsigned, unsigned> &) {};

  matrix_free_context.mf.loop(cell_op, face_op, boundary_face_op, indicator, solution);

  return VectorTools::convert_matrix_free_cell_aligned_vector_to_vector<dim, number, dim + 2>(
    matrix_free_context, indicator, tria);
}

template <int dim, typename number>
void
MeltPoolDG::AMR::JumpIndicator<dim, number>::local_apply_face(
  const dealii::MatrixFree<dim, number> &,
  dealii::AlignedVector<dealii::VectorizedArray<number>>   &dst,
  const dealii::LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned, unsigned>                      &face_range) const
{
  FEFaceIntegrator<dim, dim + 2, number> phi_m(matrix_free_context.mf,
                                               true,
                                               matrix_free_context.dof_idx,
                                               matrix_free_context.quad_idx);
  FEFaceIntegrator<dim, dim + 2, number> phi_p(matrix_free_context.mf,
                                               false,
                                               matrix_free_context.dof_idx,
                                               matrix_free_context.quad_idx);

  for (unsigned face = face_range.first; face < face_range.second; ++face)
    {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, dealii::EvaluationFlags::values);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, dealii::EvaluationFlags::values);

      dealii::VectorizedArray<number> face_area       = 0.;
      dealii::VectorizedArray<number> face_jump_error = 0.;
      for (const unsigned q : phi_m.quadrature_point_indices())
        {
          dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>> w_plus  = phi_p.get_value(q);
          dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>> w_minus = phi_m.get_value(q);

          for (int i = 1; i <= dim; ++i)
            face_jump_error += phi_m.JxW(q) * (w_minus - w_plus) * (w_minus - w_plus);

          face_area += phi_m.JxW(q);
        }
      constexpr int                   n_faces            = 2 * dim;
      dealii::VectorizedArray<number> current_cell_error = phi_m.read_cell_data(dst);
      current_cell_error += 0.25 / n_faces * std::sqrt(face_jump_error / face_area);
      phi_m.set_cell_data(dst, current_cell_error);
    }
}

template class MeltPoolDG::AMR::SSEDIndicator<1, double>;
template class MeltPoolDG::AMR::SSEDIndicator<2, double>;
template class MeltPoolDG::AMR::SSEDIndicator<3, double>;

template class MeltPoolDG::AMR::JumpIndicator<1, double>;
template class MeltPoolDG::AMR::JumpIndicator<2, double>;
template class MeltPoolDG::AMR::JumpIndicator<3, double>;
