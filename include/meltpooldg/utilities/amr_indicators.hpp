#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

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
    add_indicator(std::unique_ptr<AMRIndicatorBase<dim, number>> &&indicator, number weight = 1.);

    /**
     * Computes the composition of the internally stored indicators weighted with the corresponding
     * weighting factors. All indicators used must have been added with the add_indicator() function
     * before.
     *
     * @param tria Triangulation for which the indicator is computed.
     */
    VectorType
    compute_indicator(const dealii::Triangulation<dim> &tria) override;

    /**
     * Returns true if no indicator is stored in the object.
     */
    bool
    empty() const;

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
   * @tparam dim Spatial dimension of the problem.
   * @tparam number Scalar type of the solution vector and underlying matrix-free data structures
   * (e.g., float, double).
   * @tparam n_dof_components Number of field components provided by the finite element at each
   * point. This must match the number of components returned by dealii::FEEvaluation::get_value()
   * for the matrix-free object passed to the constructor.
   * @tparam n_indicator_components Number of components (out of the @p n_dof_components available)
   * that are used in the indicator computation. A custom extraction function maps the full set of
   * DoF values to these selected components.
   *
   * @note Indicator can only be used for fe degree > 1.
   */
  template <int dim, typename number, int n_dof_components, int n_indicator_components>
  class SSEDIndicator : public AMRIndicatorBase<dim, number>
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    using ExtractIndicatorDofValuesFn =
      std::function<dealii::Tensor<1, n_indicator_components, VectorizedArrayType>(
        const dealii::Tensor<1, n_dof_components, VectorizedArrayType> &)>;

  public:
    /**
     * Constructor.
     *
     * Sets up the SSED indicator by storing references to the matrix-free context and solution
     * vector, as well as the extraction function that specifies which components of the solution
     * should be considered in the indicator calculation. Also initializes the internal element
     * transformation matrix.
     *
     * @param matrix_free_context Matrix-free context associated with the provided solution vector.
     * @param solution Solution vector from which the indicator is computed.
     * @param extract_indicator_dof_values A function object used to extract the subset of DoF
     * values relevant for the indicator. The function receives, for each vectorized set of
     * quadrature points, the full @p n_dof_components tensor of values from
     * dealii::FEEvaluation::get_value(), and must return the @p n_indicator_components values to be
     * used by the indicator. The function may also modify or combine the values if needed.
     * @param fe_index Index of the finite element associated with the matrix-free context.
     * @param fe_degree Polynomial degree of the finite element shape functions.
     */
    SSEDIndicator(const MatrixFreeContext<dim, number> matrix_free_context,
                  const VectorType                    &solution,
                  const ExtractIndicatorDofValuesFn    extract_indicator_dof_values,
                  const unsigned                       fe_index,
                  const unsigned                       fe_degree);

    /**
     * Computes the indicator as discussed in the class description.
     *
     * @param tria Triangulation for which the indicator is computed.
     */
    dealii::Vector<number>
    compute_indicator(const dealii::Triangulation<dim> &tria) override;

  protected:
    /**
     * The actual logic of the ssed indicator performed in each cell in a matrix free loop.
     *
     * @note This is protected such that a combined indicator with additional face and/or boundary face
     * loop can be directly derived from this class.
     */
    void
    local_apply_cell(const dealii::MatrixFree<dim, number> &,
                     dealii::AlignedVector<VectorizedArrayType> &dst,
                     const VectorType                           &src,
                     const std::pair<unsigned, unsigned>        &cell_range);

  private:
    /// Matrix free context used to compute the solution.
    const MatrixFreeContext<dim, number> matrix_free_context;

    /// Reference to the solution of the field of interest.
    const VectorType &solution;

    /// Function type used to extract or manipulate the subset of DoF values needed by the
    /// indicator. This function needs to be provided by the user.
    ExtractIndicatorDofValuesFn extract_indicator_dof_values;

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
   * @tparam dim Spatial dimension of the problem.
   * @tparam number Scalar type of the solution vector and underlying matrix-free data structures
   * (e.g., float, double).
   * @tparam n_dof_components Number of field components provided by the finite element at each
   * point. This must match the number of components returned by dealii::FEEvaluation::get_value()
   * for the matrix-free object passed to the constructor.
   * @tparam n_indicator_components Number of components (out of the @p n_dof_components available)
   * that are used in the indicator computation. A custom extraction function maps the full set of
   * DoF values to these selected components.
   *
   * @note Indicator can only be used for DG.
   */
  template <int dim, typename number, int n_mf_components = dim, int n_relevant_components = dim>
  class JumpIndicator : public AMRIndicatorBase<dim, number>
  {
  public:
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using ExtractIndicatorDofValuesFn =
      std::function<dealii::Tensor<1, n_relevant_components, VectorizedArrayType>(
        const dealii::Tensor<1, n_mf_components, VectorizedArrayType> &)>;

    /**
     * Constructor.
     *
     * Initializes the indicator by storing references to the provided data.
     *
     * @param matrix_free_context Matrix-free context associated with the provided solution vector.
     * @param solution Solution vector from which the indicator is computed.
     * @param extract_indicator_dof_values A function object used to extract the subset of DoF
     * values relevant for the indicator. The function receives, for each vectorized set of
     * quadrature points, the full @p n_dof_components tensor of values from
     * dealii::FEEvaluation::get_value(), and must return the @p n_indicator_components values to be
     * used by the indicator. The function may also modify or combine the values if needed.
     * @param fe_index Index of the finite element associated with the matrix-free context.
     * @param fe_degree Polynomial degree of the finite element shape functions.
     */
    JumpIndicator(const MatrixFreeContext<dim, number> matrix_free_context,
                  const VectorType                    &solution,
                  const ExtractIndicatorDofValuesFn    extract_indicator_dof_values,
                  const unsigned                       fe_index = 0);

    /**
     * Computes the indicator as discussed in the class description.
     *
     * @param tria Triangulation for which the indicator is computed.
     */
    dealii::Vector<number>
    compute_indicator(const dealii::Triangulation<dim> &tria) override;

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
                     const std::pair<unsigned, unsigned>        &face_range) const;

  private:
    /// Matrix free object used to compute the solution.
    const MatrixFreeContext<dim, number> matrix_free_context;

    /// Reference to the solution of the field of interest.
    const VectorType &solution;

    /// Function type used to extract or manipulate the subset of DoF values needed by the
    /// indicator. This function needs to be provided by the user.
    ExtractIndicatorDofValuesFn extract_indicator_dof_values;

    /// Relevant fe index for the fe object of the dof handler in the matrix free object.
    const unsigned fe_index;
  };
} // namespace MeltPoolDG::AMR
