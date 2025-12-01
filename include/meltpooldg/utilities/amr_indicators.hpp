#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include "meltpooldg/utilities/vector_tools.hpp"
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace MeltPoolDG::AMR
{
  /**
   * Base class for mesh refinement indicators used with the indicator composite class.
   *
   * @note When using the AMR implementation of MeltPoolDG, it is not strictly required to derive
   * indicators from this class.
   */
  template <int dim, typename number>
  class AMRIndicatorBase
  {
  public:
    /**
     * Core function of the indicator.
     * For the given triangulation, returns a vector of indicator values, where each entry
     * corresponds to a cell with the same active cell index. Larger values indicate a higher need
     * for refinement, while smaller values indicate a lower need.
     *
     * @param tria Triangulation for which the refinement indicator is computed.
     */
    virtual dealii::Vector<number>
    compute_indicator(const dealii::Triangulation<dim> &tria) = 0;

    virtual ~AMRIndicatorBase() = default;
  };

  /**
   * A composite of indicators based on a specified binary operation. This indicator composite
   * stores a collection of user defined indicators and computes a reduced indicator based on the
   * specified binary operator. Optionally, the indicators can also be weighted.
   */
  template <int dim, typename number, template <typename> class BinaryOp = std::plus>
  class BinaryOpIndicatorComposite : public AMRIndicatorBase<dim, number>
  {
    using VectorType = dealii::Vector<number>;

  public:
    /**
     * Adds the provided indicator to the indicator composition.
     *
     * @param indicator Indicator to be added to the composition.
     * @param weight Optional weighting factor for the indicator during the reduction.
     */
    void
    add_indicator(std::unique_ptr<AMRIndicatorBase<dim, number>> &&indicator, number weight = 1.)
    {
      indicators_and_weights.emplace_back(std::move(indicator), weight);
    }

    /**
     * Computes the composition of the internally stored indicators weighted with the corresponding
     * weighting factors. All indicators used must have been added with the add_indicator() function
     * before.
     *
     * @param tria Triangulation for which the indicator is computed.
     */
    VectorType
    compute_indicator(const dealii::Triangulation<dim> &tria) override
    {
      AssertThrow(not indicators_and_weights.empty(),
                  dealii::ExcMessage(
                    "No AMR indicator has been added to the indicator composite."));

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
            indicator[j] = BinaryOp<number>()(indicator[j],
                                              indicators_and_weights[i].weight * temp_indicator[j]);
        }

      return indicator;
    }

    /**
     * Returns true if no indicator is stored in the object.
     */
    bool
    empty() const
    {
      return indicators_and_weights.empty();
    }

  private:
    struct IndicatorAndWeight
    {
      /// Refinement indicator
      std::unique_ptr<AMRIndicatorBase<dim, number>> indicator;

      /// Scaling weight
      number weight;
    };

    /// Vector storing the combination of refinement indicator and weights.
    std::vector<IndicatorAndWeight> indicators_and_weights;
  };

  /**
   * SSED indicator based on the description in
   *
   * F. Basile et al, A high-order h-adaptive discontinuous Galerkin method for unstructured grids
   * based on a posteriori error estimation. https://arc.aiaa.org/doi/10.2514/6.2021-1696
   *
   * The idea is that one estimates the error with the largest mode in the solution. The largest
   * mode is isolated and used as an error indicator normalized by the volume of the cell.
   *
   * @note Indicator can only be used for fe degree > 1.
   */
  template <int dim, typename number, int n_mf_components = dim, int n_relevant_components = dim>
  class SSEDIndicator : public AMRIndicatorBase<dim, number>
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using GetRelevantDofValuesFunctionType =
      std::function<dealii::Tensor<1, n_relevant_components, VectorizedArrayType>(
        const dealii::Tensor<1, n_mf_components, VectorizedArrayType> &)>;


  public:
    /**
     * Constructor.
     *
     * Initializes the indicator by storing references to the provided data and
     * setting up the element transformation matrix.
     *
     * @param matrix_free Matrix-free object used to compute the solution field on which the indicator
     * is based.
     * @param solution Solution field for which the indicator is evaluated.
     * @param fe_index FE index associated with the computation.
     * @param fe_degree Polynomial degree of the finite element shape functions.
     */
    SSEDIndicator(const MatrixFreeContext<dim, number>                      matrix_free_context,
                  const dealii::LinearAlgebra::distributed::Vector<number> &solution,
                  GetRelevantDofValuesFunctionType                          get_relevant_dof_values,
                  const unsigned                                            fe_index,
                  const unsigned                                            fe_degree)
      : matrix_free_context(matrix_free_context)
      , solution(solution)
      , get_relevant_dof_values(get_relevant_dof_values)
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

    /**
     * Computes the indicator as discussed in the class description.
     *
     * @param tria Triangulation for which the indicator is computed.
     */
    dealii::Vector<number>
    compute_indicator(const dealii::Triangulation<dim> &tria) override
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

  protected:
    /**
     * The actual logic of the ssed indicator performed in each cell in a matrix free loop.
     *
     * @note This is protected such that a combined indicator with additional face and/or boundary face
     * loop can be directly derived from this class.
     */
    void
    local_apply_cell(const dealii::MatrixFree<dim, number> &,
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
          const unsigned                  fe_degree =
            matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx)
              .get_fe(phi_dgq.get_active_fe_index())
              .degree;
          // transform cell solution values from modal basis to modal basis
          dealii::internal::FEEvaluationImplBasisChange<
            dealii::internal::evaluate_general,
            dealii::internal::EvaluatorQuantity::value,
            dim,
            0,
            0>::template do_forward<dealii::VectorizedArray<number>,
                                    dealii::VectorizedArray<number>>(n_mf_components,
                                                                     linearized_matrix,
                                                                     phi_dgq.begin_dof_values(),
                                                                     phi_leg.begin_dof_values(),
                                                                     fe_degree + 1,
                                                                     fe_degree + 1);

          // only keep highest modal basis
          for (unsigned i = 0; i < std::pow(fe_degree, dim); ++i)
            phi_leg.submit_dof_value(dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>(),
                                     i);

          phi_leg.evaluate(dealii::EvaluationFlags::values);

          // integrate over cell
          for (const unsigned int q : phi_dgq.quadrature_point_indices())
            {
              const auto u_q = get_relevant_dof_values(phi_leg.get_value(q));
              ssed_error += phi_dgq.JxW(q) * u_q * u_q;
              cell_volume += phi_dgq.JxW(q);
            }

          ssed_error /= cell_volume;
          phi_dgq.set_cell_data(dst, std::sqrt(ssed_error));
        }
    }

  private:
    /// Matrix free context used to compute the solution.
    const MatrixFreeContext<dim, number> matrix_free_context;

    /// Reference to the solution of the field of interest.
    const VectorType &solution;

    ///
    GetRelevantDofValuesFunctionType get_relevant_dof_values;

    /// Relevant fe index for the fe object of the dof handler in the matrix free object.
    const unsigned fe_index;

    dealii::AlignedVector<VectorizedArrayType> linearized_matrix;
  };

  /**
   * Jump indicator based on the description in
   *
   * F. Basile et al, A high-order h-adaptive discontinuous Galerkin method for unstructured grids
   * based on a posteriori error estimation. https://arc.aiaa.org/doi/10.2514/6.2021-1696
   *
   * Assuming a continuous exact solution, the idea is to express the error in terms of the jump
   * between elements normalized by the corresponding face area.
   *
   * @note Indicator can only be used for DG.
   */
  template <int dim, typename number, int n_mf_components = dim, int n_relevant_components = dim>
  class JumpIndicator : public AMRIndicatorBase<dim, number>
  {
  public:
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    using VectorType                       = dealii::LinearAlgebra::distributed::Vector<number>;
    using GetRelevantDofValuesFunctionType = std::function<dealii::Tensor<
      1,
      n_relevant_components,
      VectorizedArrayType>(const dealii::Tensor<1, n_mf_components, VectorizedArrayType> &)>;

    /**
     * Constructor.
     *
     * Initializes the indicator by storing references to the provided data.
     *
     * @param matrix_free Matrix-free context used to compute the solution field on which the indicator
     * is based.
     * @param solution Solution field for which the indicator is evaluated.
     * @param dof_idx Dof index relevant to the matrix-free object.
     */
    JumpIndicator(const MatrixFreeContext<dim, number> matrix_free_context,
                  const VectorType                    &solution,
                  GetRelevantDofValuesFunctionType     get_relevant_dof_values,
                  const unsigned                       fe_index = 0)
      : matrix_free_context(matrix_free_context)
      , solution(solution)
      , get_relevant_dof_values(get_relevant_dof_values)
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

    /**
     * Computes the indicator as discussed in the class description.
     *
     * @param tria Triangulation for which the indicator is computed.
     */
    dealii::Vector<number>
    compute_indicator(const dealii::Triangulation<dim> &tria) override
    {
      using LocalOpType = std::function<void(const dealii::MatrixFree<dim, number>      &mf,
                                             dealii::AlignedVector<VectorizedArrayType> &indicator,
                                             const VectorType                           &solution,
                                             const std::pair<unsigned, unsigned> &cell_range)>;

      dealii::AlignedVector<VectorizedArrayType> indicator;
      matrix_free_context.mf.initialize_cell_data_vector(indicator);

      LocalOpType cell_op = [](const dealii::MatrixFree<dim, number> &,
                               dealii::AlignedVector<VectorizedArrayType> &,
                               const VectorType &,
                               const std::pair<unsigned, unsigned> &) {};

      LocalOpType face_op = std::bind_front(
        &JumpIndicator<dim, number, n_mf_components, n_relevant_components>::local_apply_face,
        this);

      LocalOpType boundary_face_op = [](const dealii::MatrixFree<dim, number> &,
                                        dealii::AlignedVector<VectorizedArrayType> &,
                                        const VectorType &,
                                        const std::pair<unsigned, unsigned> &) {};

      matrix_free_context.mf.loop(cell_op, face_op, boundary_face_op, indicator, solution);

      return VectorTools::
        convert_matrix_free_cell_aligned_vector_to_vector<dim, number, n_mf_components>(
          matrix_free_context, indicator, tria);
    }

  protected:
    /**
     * The actual logic of the jump indicator performed in each cell in a matrix free loop.
     *
     * @note This is protected such that a combined indicator with additional cell or boundary face
     * loop can be directly derived from this class.
     */
    void
    local_apply_face(const dealii::MatrixFree<dim, number> &,
                     dealii::AlignedVector<VectorizedArrayType> &dst,
                     const VectorType                           &src,
                     const std::pair<unsigned, unsigned>        &face_range) const
    {
      FEFaceIntegrator<dim, n_mf_components, number> phi_m(matrix_free_context.mf,
                                                           true,
                                                           matrix_free_context.dof_idx,
                                                           matrix_free_context.quad_idx);
      FEFaceIntegrator<dim, n_mf_components, number> phi_p(matrix_free_context.mf,
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
              dealii::Tensor<1, n_relevant_components, VectorizedArrayType> w_plus =
                get_relevant_dof_values(phi_p.get_value(q));
              dealii::Tensor<1, n_relevant_components, VectorizedArrayType> w_minus =
                get_relevant_dof_values(phi_m.get_value(q));

              face_jump_error += phi_m.JxW(q) * (w_minus - w_plus) * (w_minus - w_plus);

              face_area += phi_m.JxW(q);
            }
          constexpr int       n_faces            = 2 * dim;
          VectorizedArrayType current_cell_error = phi_m.read_cell_data(dst);
          current_cell_error += 0.25 / n_faces * std::sqrt(face_jump_error / face_area);
          phi_m.set_cell_data(dst, current_cell_error);
        }
    }

  private:
    /// Matrix free object used to compute the solution.
    const MatrixFreeContext<dim, number> matrix_free_context;

    /// Reference to the solution of the field of interest.
    const VectorType &solution;

    ///
    GetRelevantDofValuesFunctionType get_relevant_dof_values;

    /// Relevant fe index for the fe object of the dof handler in the matrix free object.
    const unsigned fe_index;
  };
} // namespace MeltPoolDG::AMR
