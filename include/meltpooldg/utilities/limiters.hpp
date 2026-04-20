#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <boost/container/small_vector.hpp>

#include <array>
#include <functional>
#include <vector>

namespace MeltPoolDG::Utilities
{
  /**
   * An enumeration of the available limiters.
   */
  BETTER_ENUM(LimiterType, char, tvd_minmod, tvb_minmod)

  /**
   * A struct to hold the data for the limiters.
   */
  template <typename number>
  struct LimiterData
  {
    /// Boolean flag indicating whether to apply the limiter or not.
    bool apply_limiter = false;

    /// The TVB constant used in the TVB minmod limiter.
    number tvb_constant = 1.0;

    /// The type of limiter to apply.
    LimiterType type = LimiterType::tvd_minmod;

    /**
     * Add the limiter parameters to the parameter handler.
     *
     * @param prm The parameter handler to which the limiter parameters will be added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  /**
   * This function computes the component-wise minmod operator over all tensors in the
   * given container. The standard minmod calculation for each component index $i$ is:
   * - If all candidate values share the same sign, it returns the value with the smallest absolute
   *   magnitude.
   * - If there is any sign change (indicating a local extremum), it returns zero to damp
   *   oscillations.
   *
   * @tparam Container The type of the container holding the values. This can be any standard
   * container (e.g., `std::vector`, `std::array`, `boost::container::small_vector`) that supports
   * iterating over it and contains elements of type `TensorType`.
   *
   * @param values The container of values for which to compute the minmod.
   *
   * @return The minmod of the given values.
   */
  template <int n_components, typename TensorType, typename Container>
  TensorType
  tvd_minmod(const Container &values);

  /**
   * Computes the Total Variation Bounded (TVB) modified minmod of a container of tensors. The TVB
   * minmod variant relaxes the strict TVD minmod criteria near smooth extrema to avoid unnecessary
   * clipping of physical peaks.  For each component, if the initial candidate slope (`values[0]`)
   * is bounded by the threshold $M \cdot \Delta x^2$ (where $M$ is `tvb_constant` and $\Delta x$ is
   * `cell_size`), the slope is considered smooth and left entirely unmodified. If it exceeds this
   * threshold, the function falls back to the standard `tvd_minmod` routine for that component.
   *
   * @tparam Container The type of the container holding the values. This can be any standard
   * container (e.g., `std::vector`, `std::array`, `boost::container::small_vector`) that supports
   * iterating over it and contains elements of type `TensorType`.
   *
   * @param values The container of tensors to evaluate. The first element (`values[0]`)
   * represents the current cell slope on the cell of interest.
   * @param tvb_constant The tuning constant $M > 0$ determining the strictness of the bound.
   * @param cell_size The characteristic mesh size of the local cell block, e.g. the mesh size in
   * the direction of the slope being limited.
   *
   * @return A tensor containing the component-wise TVB limited slopes.
   */
  template <typename number, int n_components, typename TensorType, typename Container>
  TensorType
  tvb_minmod(const Container &values, const number tvb_constant, const number cell_size);

  /**
   * This function computes the limited slopes for a given cell using a minmod-type limiter. It
   * computes the finite difference slopes with respect to the cell averages of the current cell and
   * its neighbors in all direction separately and passes them together with the average cell
   * gradient to the appropriate minmod-type limiter function (TVD or TVB) based on the specified
   * limiter type in the limiter data struct. The resulting limited slopes are returned as an array
   * of tensors, where each tensor corresponds to a spatial direction.
   *
   * @param cell_average_values A vector containing the cell average values for all cells in the
   * domain.
   * @param cell The active cell iterator for the current cell for which the limited slopes are
   * being computed.
   * @param average_cell_gradient The average cell gradient for the current cell, i.e. a linearized
   * representation of the solution gradient in the cell.
   * @param limiter_data Struct containing the limiter type and relevant parameters for the limiter.
   *
   * @return An array of tensors containing the limited slopes for each spatial direction.
   */
  template <int dim, int n_components, typename number>
  std::array<dealii::Tensor<1, n_components, number>, dim>
  compute_minmod_type_limited_slopes(
    const std::vector<dealii::Tensor<1, n_components, number>>            &cell_average_values,
    const typename dealii::Triangulation<dim>::active_cell_iterator       &cell,
    const dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>> &average_cell_gradient,
    const LimiterData<number>                                             &limiter_data);

  /**
   * This function applies a minmod-type limiter to the solution vector in a cell-wise manner using
   * the provided matrix-free context. It basically iterates over each cell to compute the limited
   * solution based on the specified limiter type in the limiter data struct.
   * For more details see Cockburn and Shu, "The Runge-Kutta Discontinuous Galerkin Method for
   * Conservation Laws V: Multidimensional Systems", Journal of Computational Physics,
   * 141(2):199–224, 1998.
   *
   * @param mf_context Context containing a reference to the matrix-free object and relevant indices
   * for dofs and quadrature.
   * @param dst The destination vector where the limited solution will be stored.
   * @param src The source vector containing the current solution values to be limited.
   * @param limiter_data Struct containing the limiter type and relevant parameters for the limiter.
   */
  template <int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType = dealii::VectorizedArray<number>,
            typename VectorType          = dealii::LinearAlgebra::distributed::Vector<number>>
  void
  apply_minmod_type_limiter(const MatrixFreeContext<dim, number> &mf_context,
                            VectorType                           &dst,
                            const VectorType                     &src,
                            const LimiterData<number>            &limiter_data);

} // namespace MeltPoolDG::Utilities


/*********************************************************************************************
 * Function definitions
 *********************************************************************************************/
template <typename number>
void
MeltPoolDG::Utilities::LimiterData<number>::add_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("limiter");
  {
    prm.add_parameter("apply", apply_limiter, "Whether to apply a limiter.");
    prm.add_parameter("tvb constant",
                      tvb_constant,
                      "The TVB constant used in the TVB minmod limiter.");
    prm.add_parameter("type", type, "The type of limiter to apply.");
  }
  prm.leave_subsection();
}


template <int n_components, typename TensorType, typename Container>
TensorType
MeltPoolDG::Utilities::tvd_minmod(const Container &values)
{
  Assert(values.begin() != values.end(), dealii::ExcMessage("Container must not be empty."));

  TensorType result;
  for (unsigned int i = 0; i < n_components; ++i)
    {
      // Check sign consistency
      const bool all_positive =
        std::all_of(values.begin(), values.end(), [i](TensorType v) { return v[i] > 0; });
      const bool all_negative =
        std::all_of(values.begin(), values.end(), [i](TensorType v) { return v[i] < 0; });

      if (!(all_positive or all_negative))
        {
          result[i] = typename TensorType::value_type(0.);
        }
      else
        {
          // Find element with smallest absolute value
          auto it = std::min_element(values.begin(),
                                     values.end(),
                                     [i](const TensorType &a, const TensorType &b) {
                                       return std::abs(a[i]) < std::abs(b[i]);
                                     });

          result[i] = (*it)[i];
        }
    }

  return result;
}


template <typename number, int n_components, typename TensorType, typename Container>
TensorType
MeltPoolDG::Utilities::tvb_minmod(const Container &values,
                                  const number     tvb_constant,
                                  const number     cell_size)
{
  Assert(values.begin() != values.end(), dealii::ExcMessage("Container must not be empty."));

  TensorType                     result;
  std::array<bool, n_components> below_limit_mask;
  for (unsigned int i = 0; i < n_components; ++i)
    {
      if (std::abs(values[0][i]) <= tvb_constant * cell_size * cell_size)
        {
          below_limit_mask[i] = true;
        }
      else
        {
          below_limit_mask[i] = false;
        }
    }

  if (std::all_of(below_limit_mask.begin(), below_limit_mask.end(), [](bool v) { return v; }))
    {
      for (unsigned int i = 0; i < n_components; ++i)
        result[i] = values[0][i];
    }
  else
    {
      result = tvd_minmod<n_components, TensorType, Container>(values);
      for (unsigned int i = 0; i < n_components; ++i)
        {
          if (below_limit_mask[i])
            result[i] = values[0][i];
        }
    }

  return result;
}


template <int dim, int n_components, typename number>
std::array<dealii::Tensor<1, n_components, number>, dim>
MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(
  const std::vector<dealii::Tensor<1, n_components, number>>            &cell_average_values,
  const typename dealii::Triangulation<dim>::active_cell_iterator       &cell,
  const dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>> &average_cell_gradient,
  const MeltPoolDG::Utilities::LimiterData<number>                      &limiter_data)
{
  std::array<dealii::Tensor<1, n_components, number>, dim> limited_slopes;

  for (unsigned int d = 0; d < dim; ++d)
    {
      boost::container::small_vector<dealii::Tensor<1, n_components, number>, 3>
        minmod_input_values;
      minmod_input_values.emplace_back();
      for (unsigned int i = 0; i < n_components; ++i)
        minmod_input_values[0][i] = average_cell_gradient[i][d];
      for (unsigned int face_no = 0; face_no < 2; ++face_no)
        {
          const auto neighbor = cell->neighbor(2 * d + face_no);

          if (!cell->at_boundary(2 * d + face_no) and neighbor->is_active())
            {
              // TODO: What happens in case of local refinement?
              AssertThrow(
                neighbor->level() == cell->level(),
                dealii::ExcMessage(
                  "The current implementation of the minmod-type limiters does not support local mesh refinement."));

              auto   vector_to_neighbor   = cell->center() - neighbor->center();
              number distance_to_neighbor = vector_to_neighbor.norm();

              if (face_no == 0)
                {
                  minmod_input_values.push_back(
                    (cell_average_values[cell->active_cell_index()] -
                     cell_average_values[neighbor->active_cell_index()]) /
                    distance_to_neighbor);
                }
              else
                {
                  minmod_input_values.push_back(
                    (cell_average_values[neighbor->active_cell_index()] -
                     cell_average_values[cell->active_cell_index()]) /
                    distance_to_neighbor);
                }
            }
        }

      switch (limiter_data.type)
        {
          case LimiterType::tvd_minmod:
            limited_slopes[d] = tvd_minmod<
              n_components,
              dealii::Tensor<1, n_components, number>,
              boost::container::small_vector<dealii::Tensor<1, n_components, number>, 3>>(
              minmod_input_values);
            break;
          case LimiterType::tvb_minmod:
            limited_slopes[d] = tvb_minmod<
              number,
              n_components,
              dealii::Tensor<1, n_components, number>,
              boost::container::small_vector<dealii::Tensor<1, n_components, number>, 3>>(
              minmod_input_values, limiter_data.tvb_constant, cell->extent_in_direction(d));
            break;
          default:
            AssertThrow(false, dealii::ExcMessage("Unknown limiter type."));
        }
    }
  return limited_slopes;
}


template <int dim,
          int n_components,
          typename number,
          typename VectorizedArrayType,
          typename VectorType>
void
MeltPoolDG::Utilities::apply_minmod_type_limiter(
  const MatrixFreeContext<dim, number>             &mf_context,
  VectorType                                       &dst,
  const VectorType                                 &src,
  const MeltPoolDG::Utilities::LimiterData<number> &limiter_data)
{
  if (not limiter_data.apply_limiter)
    return;

  AssertThrow(mf_context.mf.get_dof_handler(mf_context.dof_idx).get_fe().degree == 1,
              dealii::ExcMessage(
                "The minmod type limiter currently only supports linear elements (degree 1)."));

  std::vector<dealii::Tensor<1, n_components, number>> cell_average_values =
    compute_cell_average_quantities<dim, number, n_components>(mf_context, src);

  // The cell loop lambda that will be passed to the matrix-free object to apply the limiter in
  // a cell-wise manner. For each cell, the lambda computes the limited slopes at quadrature
  // points and applies them to compute the limited solution values at quadrature points, which
  // are then transformed back to the nodal basis and stored in the destination vector.
  const std::function<void(const dealii::MatrixFree<dim, number, VectorizedArrayType> &,
                           VectorType &,
                           const VectorType &,
                           const std::pair<unsigned int, unsigned int> &)>
    limit_loop = [&](const dealii::MatrixFree<dim, number, VectorizedArrayType> &matrix_free,
                     VectorType                                                 &dst,
                     const VectorType                                           &src,
                     const std::pair<unsigned int, unsigned int>                &cell_range) {
      FECellIntegrator<dim, n_components, number> fe_cell_integrator(matrix_free,
                                                                     mf_context.dof_idx,
                                                                     mf_context.quad_idx);

      dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
        fe_cell_integrator);

      std::vector<VectorizedArrayType> limited_values(
        n_components * fe_cell_integrator.quadrature_point_indices().size());

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          fe_cell_integrator.reinit(cell);
          fe_cell_integrator.gather_evaluate(src,
                                             dealii::EvaluationFlags::values |
                                               dealii::EvaluationFlags::gradients);

          const auto &cells = cells_in_cell_batch(mf_context.mf, cell);

          dealii::Tensor<1, n_components, dealii::Tensor<1, dim, VectorizedArrayType>>
            limited_slopes;

          dealii::Point<dim, VectorizedArrayType> cell_centers;
          for (unsigned int i = 0; i < mf_context.mf.n_active_entries_per_cell_batch(cell); ++i)
            for (unsigned int d = 0; d < dim; ++d)
              cell_centers[d][i] = cells[i]->center()[d];

          dealii::Tensor<1, n_components, VectorizedArrayType> cell_average_values_cell;
          for (unsigned int i = 0; i < mf_context.mf.n_active_entries_per_cell_batch(cell); ++i)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                cell_average_values_cell[c][i] =
                  cell_average_values[cells[i]->active_cell_index()][c];
            }

          for (const unsigned int q : fe_cell_integrator.quadrature_point_indices())
            {
              dealii::Tensor<1, dim, VectorizedArrayType> vector_to_dof =
                fe_cell_integrator.quadrature_point(q) - cell_centers;

              // The cell loop lambda that will be passed to the matrix-free object to apply the
              // limiter in a cell-wise manner. For each cell, the lambda computes the limited
              // slopes at quadrature points and applies them to compute the limited solution
              // values at quadrature points, which are then transformed back to the nodal basis
              // and stored in the destination vector.
              if (q == 0)
                {
                  for (unsigned int lane = 0;
                       lane < mf_context.mf.n_active_entries_per_cell_batch(cell);
                       ++lane)
                    {
                      dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>>
                        gradient_value_lane;
                      for (unsigned int c = 0; c < n_components; ++c)
                        for (unsigned int d = 0; d < dim; ++d)
                          gradient_value_lane[c][d] =
                            fe_cell_integrator.get_gradient(q)[c][d][lane];

                      std::array<dealii::Tensor<1, n_components, number>, dim> limited_slope_lane =
                        compute_minmod_type_limited_slopes<dim, n_components, number>(
                          cell_average_values, cells[lane], gradient_value_lane, limiter_data);

                      for (unsigned int c = 0; c < n_components; ++c)
                        for (unsigned int d = 0; d < dim; ++d)
                          limited_slopes[c][d][lane] = limited_slope_lane[d][c];
                    }
                }

              for (unsigned int c = 0; c < n_components; ++c)
                limited_values[c * fe_cell_integrator.quadrature_point_indices().size() + q] =
                  cell_average_values_cell[c] + limited_slopes[c] * vector_to_dof;
            }

          inverse.transform_from_q_points_to_basis(n_components,
                                                   limited_values.data(),
                                                   fe_cell_integrator.begin_dof_values());
          fe_cell_integrator.set_dof_values(dst);
        }
    };

  mf_context.mf.cell_loop(limit_loop, dst, src, true);
}
