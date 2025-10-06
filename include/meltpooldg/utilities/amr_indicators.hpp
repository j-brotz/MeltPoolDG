#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/fe_integrator.hpp>

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
  template <int dim, typename number>
  class SSEDIndicator : public AMRIndicatorBase<dim, number>
  {
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
     * @param dof_idx Dof index relevant to the matrix-free object.
     * @param quad_idx Quadrature point index relevant to the matrix-free object.
     * @param fe_index FE index associated with the computation.
     * @param fe_degree Polynomial degree of the finite element shape functions.
     */
    SSEDIndicator(const dealii::MatrixFree<dim, number>                    &matrix_free,
                  const dealii::LinearAlgebra::distributed::Vector<number> &solution,
                  const unsigned                                            dof_idx,
                  const unsigned                                            quad_idx,
                  const unsigned                                            fe_index,
                  const unsigned                                            fe_degree);

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
                     dealii::AlignedVector<dealii::VectorizedArray<number>>   &dst,
                     const dealii::LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned, unsigned>                      &cell_range);

  private:
    /// Matrix free object used to compute the solution.
    const dealii::MatrixFree<dim, number> &matrix_free;

    /// Reference to the solution of the field of interest.
    const dealii::LinearAlgebra::distributed::Vector<number> &solution;

    /// Relevant dof index in the matrix free object.
    const unsigned mf_dof_idx;

    /// Relevant quad index in the matrix free object.
    const unsigned mf_quad_idx;

    /// Relevant fe index for the fe object of the dof handler in the matrix free object.
    const unsigned fe_index;

    dealii::AlignedVector<dealii::VectorizedArray<number>> linearized_matrix;
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
  template <int dim, typename number>
  class JumpIndicator : public AMRIndicatorBase<dim, number>
  {
  public:
    /**
     * Constructor.
     *
     * Initializes the indicator by storing references to the provided data.
     *
     * @param matrix_free Matrix-free object used to compute the solution field on which the indicator
     * is based.
     * @param solution Solution field for which the indicator is evaluated.
     * @param dof_idx Dof index relevant to the matrix-free object.
     * @param quad_idx Quadrature point index relevant to the matrix-free object.
     * @param fe_index FE index associated with the computation.
     */
    JumpIndicator(const dealii::MatrixFree<dim, number>                    &matrix_free,
                  const dealii::LinearAlgebra::distributed::Vector<number> &solution,
                  const unsigned                                            dof_idx,
                  const unsigned                                            quad_idx,
                  const unsigned                                            fe_index);

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
                     dealii::AlignedVector<dealii::VectorizedArray<number>>   &dst,
                     const dealii::LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned, unsigned>                      &face_range) const;

  private:
    /// Matrix free object used to compute the solution.
    const dealii::MatrixFree<dim, number> &matrix_free;

    /// Reference to the solution of the field of interest.
    const dealii::LinearAlgebra::distributed::Vector<number> &solution;

    /// Relevant dof index in the matrix free object.
    const unsigned mf_dof_idx;

    /// Relevant quad index in the matrix free object.
    const unsigned mf_quad_idx;

    /// Relevant fe index for the fe object of the dof handler in the matrix free object.
    const unsigned fe_index;
  };
} // namespace MeltPoolDG::AMR
