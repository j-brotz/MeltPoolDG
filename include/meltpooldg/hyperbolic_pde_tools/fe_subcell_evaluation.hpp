#pragma once

#include <deal.II/base/std_cxx20/iota_view.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

namespace MeltPoolDG::HyperbolicPDETools
{
  /**
   * This class provides a convenient interface for evaluating subcell values on a single finite
   * element cell. The class is designed with finite volume subcell limiters in mind, where a
   * troubled DG cell is split into a number of smaller subcells, on which a time step is
   * performed using a more robust finite volume scheme.
   *
   * The idea is illustrated below for a two-dimensional DG cell with `n_subcells_1d == 4`, i.e.
   * split into sixteen subcells. The number in each subcell is its `subcell_index`, as used by
   * get_subcell_value() and get_subcell_neighbor_value() -- the same lexicographic ordering as
   * dealii::FEEvaluation::quadrature_point_indices(), with the x-index varying fastest:
   *
   * @verbatim
   *    y
   *    ^
   *    |   +------+------+------+------+
   *    |   |  12  |  13  |  14  |  15  |
   *    |   +------+------+------+------+
   *    |   |   8  |   9  |  10  |  11  |
   *    |   +------+------+------+------+
   *    |   |   4  |   5  |   6  |   7  |
   *    |   +------+------+------+------+
   *    |   |   0  |   1  |   2  |   3  |
   *    |   +------+------+------+------+
   *    +----------------------------------> x
   * @endverbatim
   *
   * In general the interface of the class is designed in a similar way to dealii::FEEvaluation.
   *
   * @note **Important**: The class requires that the cell batches used by the underlying
   * dealii::MatrixFree object  are ordered such that all cells of a given cell batch share the same
   * face type at the same face index. This means that e.g. a cell whose faces 0 and 2 ly at a
   * domain boundary cannot be in the same cell batch as a cell whose faces 0 and 2 are interior
   * faces.
   *
   * @note **Important**: Currently cartesian cells are assumed, i.e. the subcells are assumed to be
   * aligned with the coordinate axes.
   */
  template <int dim, int n_components, typename number>
  class FESubcellEvaluation
  {
    using value_type =
      std::conditional_t<n_components == 1,
                         dealii::VectorizedArray<number>,
                         dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>>;

  public:
    FESubcellEvaluation(const dealii::MatrixFree<dim, number> &matrix_free,
                        const unsigned int                     dof_no,
                        const unsigned int                     quad_no);
    void
    reinit(const unsigned int cell_batch_index);

    void
    gather_evaluate(
      const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
      std::function<value_type(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                               dealii::types::boundary_id,
                               const value_type &)>             get_boundary_value = {});

    /**
     * Return the value of the solution on the subcell with the given index. This value can be
     * interpreted as the average value of the solution on the corresponding subcell.
     *
     * @param subcell_index The index of the subcell, in lexicographic order, with the x-index
     * varying fastest. The index must be in the range `[0, n_subcells_1d^dim)`.
     */
    value_type
    get_subcell_value(const unsigned int subcell_index) const;

    /**
     * This function returns the value of the solution on a neighboring subcell of the subcell with
     * the subcell index `subcell_index`. The neighboring subcell for which the value is returned is
     * specified by the face number `face_no` of the current subcell at which the neighbor is
     * located. For the convention of face numbers, see
     * dealii::GeometryInfo::face_to_cell_orientation(). The returned value can be interpreted as
     * the average value of the solution on the neighboring subcell.
     *
     * @param subcell_index The index of the subcell, in lexicographic order, with the x-index
     * varying fastest. The index must be in the range `[0, n_subcells_1d^dim)`.
     * @param face_no The face number of the subcell at which the neighboring subcell is located.
     */
    value_type
    get_subcell_neighbor_value(const unsigned int subcell_index, const unsigned int face_no) const;

    /**
     * Submits a value for the subcell with the given index. The value is stored internally but is
     * not written to the same place from which get_subcell_value() and get_subcell_neighbor_value()
     * read the values. Hence, contrary to the functionality of
     * dealii::FEEvaluation::submit_value(), this function does not overwrite the value of the
     * subcell with the given index, and it is safe to call get_subcell_value() and
     * get_subcell_neighbor_value() with the same subcell index after calling this function.
     *
     * @param subcell_index The index of the subcell, in lexicographic order to which the new value
     * is submitted.
     * @param value The value to be submitted.
     */
    void
    submit_subcell_value(const unsigned int subcell_index, const value_type &value);

    void
    set_dof_values(dealii::LinearAlgebra::distributed::Vector<number> &dst);

    dealii::std_cxx20::ranges::iota_view<unsigned int, unsigned int>
    subcell_indices() const;

    template <typename T>
    T
    read_cell_data(const dealii::AlignedVector<T> &data) const
    {
      return fe_cell_integrator.read_cell_data(data);
    }

    template <typename T>
    void
    set_cell_data(dealii::AlignedVector<T> &data, const T &value) const
    {
      fe_cell_integrator.set_cell_data(data, value);
    }

    dealii::VectorizedArray<number>
    subcell_size(const unsigned int subcell_index) const;

    dealii::VectorizedArray<number>
    subcell_face_size(const unsigned int subcell_index, const unsigned int face_no) const;

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    subcell_face_normal(const unsigned int subcell_index, const unsigned int face_no);

  private:
    const MatrixFreeContext<dim, number> matrix_free_context;
    unsigned int                         cell_batch_index;

    FECellIntegrator<dim, n_components, number> fe_cell_integrator;
    FEFaceIntegrator<dim, n_components, number> fe_inner_face_integrator;
    FEFaceIntegrator<dim, n_components, number> fe_outer_face_integrator;

    /**
     * The vector contains all subcell values, including one layer of neighboring subcell values on
     * each side of the cell. Subcells are addressed by a per-direction cartesian index in the
     * extended range `[-1, n_subcells_1d]`: `0 <= index < n_subcells_1d` addresses an interior
     * subcell of this cell, while `-1` / `n_subcells_1d` addresses the adjacent subcell of the
     * neighboring cell across the corresponding face (`-1` for the negative-direction neighbor,
     * `n_subcells_1d` for the positive-direction one). `subcell_values` is simply this extended
     * `(n_subcells_1d + 2)^dim` grid stored in dense lexicographic order, with the first (x) index
     * varying fastest -- i.e. index `= sum_d (coordinate_d + 1) * (n_subcells_1d + 2)^d`.
     *
     * Because the grid is dense, the "diagonal" positions where two or more directions are
     * simultaneously out of `[0, n_subcells_1d)` (corners in 2d; corners and edges in 3d) are
     * addressable and occupy a well-defined slot too, even though they are never populated: no
     * face-based finite volume flux needs a diagonal neighbor, so those entries are simply unused
     * padding, kept around so the indexing formula stays uniform and does not need to special-case
     * faces vs. corners.
     *
     * This is illustrated below for `n_subcells_1d == 2`. Each cell shows its linear index into
     * `subcell_values`; the shaded corners are the unused entries:
     *
     * @verbatim
     *                i = -1       i = 0        i = 1        i = 2
     *              +-----------+------------+------------+-----------+
     *              | (unused)  |     13     |     14     | (unused)  |  j = 2   (north face)
     *              |    12     |   (north)  |   (north)  |    15     |
     *              +-----------+------------+------------+-----------+
     *              |     8     |     9      |     10     |    11     |  j = 1
     *              |  (west)   |            |            |  (east)   |
     *              +-----------+------------+------------+-----------+
     *              |     4     |     5      |     6      |     7     |  j = 0
     *              |  (west)   |            |            |  (east)   |
     *              +-----------+------------+------------+-----------+
     *              | (unused)  |     1      |     2      | (unused)  |  j = -1  (south face)
     *              |     0     |  (south)   |  (south)   |     3     |
     *              +-----------+------------+------------+-----------+
     * @endverbatim
     *
     * i.e. subcell `(i, j)` sits at linear index `(i + 1) + 4 * (j + 1)`: the interior subcells
     * are `5, 6, 9, 10`, the face-neighbor subcells are `1, 2` (south), `13, 14` (north), `4, 8`
     * (west), `7, 11` (east), and `0, 3, 12, 15` are the unused corners.
     *
     * In order to distinguish between interior subcell indexing (5,6,9 and 10 in the example above)
     * and the complete indexing including the neighboring subcells, the terms subcell index are
     * used for the interior subcell indexing and the term padded index is used for the complete
     * indexing including the neighboring subcells.
     */
    std::vector<value_type> subcell_values;

    /**
     * A vector to store the submitted values for each subcell. The ordering is the same as above
     * but without the padding for the neighboring subcells. When the function set_dof_values() is
     * called, the values stored in this vector are used to update the values of the subcells in the
     * underlying vector.
     */
    std::vector<value_type> submitted_subcell_values;

    const unsigned int n_subcells_1d;
    const unsigned int n_padded_subcells_1d;
    const unsigned int n_subcells;

    bool
    is_subcell_face_at_cell_boundary(const unsigned int subcell_index,
                                     const unsigned int face_no) const;

    unsigned int
    subcell_index_to_padded_index(unsigned int subcell_index) const;
  };
} // namespace MeltPoolDG::HyperbolicPDETools
