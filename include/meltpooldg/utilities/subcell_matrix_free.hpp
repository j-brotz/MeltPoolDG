#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <array>
#include <memory>
#include <vector>

namespace MeltPoolDG
{
  /**
   * Index data of the subcells that is the same for all cells, analogous to the ShapeInfo of
   * dealii::MatrixFree. The subcells and the intra cell subcell faces are numbered as described in
   * FESubcellEvaluation.
   */
  template <int dim, typename number>
  struct SubcellShapeInfo
  {
    /**
     * Set up the index data for the subcells given by the 1D quadrature formula @p quadrature_1d,
     * i.e. one subcell per quadrature point.
     */
    void
    reinit(const dealii::Quadrature<1> &quadrature_1d);

    /// Number of subcells per direction.
    unsigned int n_subcells_1d = 0;

    /// Number of subcells per cell.
    unsigned int n_subcells = 0;

    /// Number of subcells adjacent to a face of the cell.
    unsigned int n_subcells_face = 0;

    /// Number of intra cell subcell faces, i.e. faces shared by two subcells of the same cell.
    unsigned int n_intra_cell_faces = 0;

    /// [face_no][q_face]: index of the subcell adjacent to the face @p face_no of the cell, in the
    /// quadrature point order of dealii::FEFaceEvaluation for faces in standard orientation.
    std::array<std::vector<unsigned int>, 2 * dim> face_to_cell_subcell_indices;

    /// [intra cell face]: index of the interior and the exterior subcell of the face.
    std::vector<std::array<unsigned int, 2>> intra_cell_face_to_subcells;

    /// [intra cell face]: direction of the normal vector of the face.
    std::vector<unsigned int> intra_cell_face_directions;

    /// Centers of the subcells on the unit interval, i.e. the 1D quadrature points.
    std::vector<number> subcell_centers_1d;

    /// Widths of the subcells on the unit interval, i.e. the 1D quadrature weights.
    std::vector<number> subcell_widths_1d;
  };

  /**
   * ## General Description
   *
   * This class precomputes the data for FESubcellEvaluation and FESubcellFaceEvaluation that does
   * not change between the calls of a matrix-free loop, analogous to dealii::MatrixFree for
   * dealii::FEEvaluation. It holds two kinds of data:
   *
   * - The index data that is the same for all cells, see SubcellShapeInfo.
   * - The neighbors across each face of the locally owned and the ghost cells held by the
   *   dealii::MatrixFree object, needed for the subcell stencils that reach over the faces of a
   *   cell.
   *
   * ## Neighbor Types
   *
   * The neighbor of a cell across a face is one of the following:
   *
   * - **held**: The neighbor is held by the dealii::MatrixFree object, either as a locally owned
   *   or as a ghost cell. Its DoF values are read with dealii::FEEvaluation::reinit() for the cell
   *   indices given in FaceNeighbors::cell_ids.
   * - **extra**: The neighbor is a ghost cell that is not held by the dealii::MatrixFree object.
   *   Its DoF values are exchanged by this class in update_ghost_values() and read with
   *   read_extra_cell_dof_values().
   * - **boundary**: The face lies at the domain boundary.
   *
   * With `hold_all_faces_to_owned_cells = true`, all face neighbors of the locally owned cells are
   * held. Extra neighbors only occur for the ghost cells, mainly at corners of the partition
   * boundary: there, the face neighbor of a ghost cell is an edge neighbor (in 2D a vertex
   * neighbor) of a locally owned cell, which dealii::MatrixFree does not hold. The face neighbors
   * of the ghost cells are needed in the face loop, since the subcell stencil on a face reaches
   * over the lateral faces of both adjacent cells.
   *
   * The face neighbors of a ghost cell that are not part of the ghost layer, i.e. artificial cells,
   * are not available. For those, no entry of FaceNeighbors is set. They are never needed, since
   * only the lateral neighbors of the cells adjacent to a face are used, which share at least an
   * edge with a locally owned cell.
   *
   * ## Requirements
   *
   * The class currently requires
   * - `MatrixFree::AdditionalData::hold_all_faces_to_owned_cells = true` with face integrals,
   * - a DG element in the DoFHandler, i.e. no DoFs shared between cells,
   * - Cartesian cells without hanging nodes,
   * - faces in standard orientation and aligned neighbors, i.e. the face @p face_no of a cell is
   *   the face `face_no ^ 1` of its neighbor, including periodic neighbors.
   *
   * reinit() must be called again whenever the dealii::MatrixFree object is reinitialized, e.g.
   * after adaptive mesh refinement.
   *
   * ## Usage Example
   *
   * In the cell loop, the face neighbors of a cell batch are read directly:
   *
   * @code
   * subcell_matrix_free.update_ghost_values(src);
   * ...
   * const SubcellMatrixFree<dim, double>::FaceNeighbors &neighbors =
   *   subcell_matrix_free.get_face_neighbors(cell_batch, face_no);
   * fe_eval_neighbor.reinit(neighbors.cell_ids);
   * fe_eval_neighbor.read_dof_values(src);
   * if (not neighbors.all_held)
   *   for (unsigned int lane = 0; lane < n_lanes; ++lane)
   *     if (neighbors.extra_cell_ids[lane] != dealii::numbers::invalid_unsigned_int)
   *       subcell_matrix_free.read_extra_cell_dof_values(neighbors.extra_cell_ids[lane],
   *                                                      lane,
   *                                                      fe_eval_neighbor.begin_dof_values());
   * @endcode
   *
   * In the face loop, the lateral neighbors of the cells adjacent to a face batch are gathered
   * from the cells given in dealii::internal::MatrixFreeFunctions::FaceToCellTopology:
   *
   * @code
   * const SubcellMatrixFree<dim, double>::FaceNeighbors neighbors =
   *   subcell_matrix_free.get_face_neighbors(matrix_free.get_face_info(face).cells_exterior,
   *                                          lateral_face_no);
   * @endcode
   */
  template <int dim, typename number>
  class SubcellMatrixFree
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;

  public:
    static constexpr unsigned int n_lanes = VectorizedArrayType::size();

    /**
     * The neighbors across one face of `n_lanes` cells. Each lane holding a cell has exactly one
     * of the entries #cell_ids, #extra_cell_ids and #boundary_ids set, unless the neighbor is not
     * available (see the class description). All entries of the other lanes are invalid.
     */
    struct FaceNeighbors
    {
      /**
       * Constructor setting all entries to invalid.
       */
      FaceNeighbors();

      /// Cell index of a held neighbor as used by dealii::MatrixFree, i.e.
      /// `cell_batch * n_lanes + lane`, and dealii::numbers::invalid_unsigned_int otherwise. The
      /// array can be passed to dealii::FEEvaluation::reinit() directly.
      std::array<unsigned int, n_lanes> cell_ids;

      /// Index of an extra neighbor for read_extra_cell_dof_values() and
      /// dealii::numbers::invalid_unsigned_int otherwise.
      std::array<unsigned int, n_lanes> extra_cell_ids;

      /// Boundary id of the face at the domain boundary and dealii::numbers::invalid_boundary_id
      /// otherwise.
      std::array<dealii::types::boundary_id, n_lanes> boundary_ids;

      /// Extent of the neighbor in the direction normal to the face, zero at the domain boundary.
      VectorizedArrayType extents;

      /// True if the neighbors of all lanes holding a cell are held, i.e. a plain
      /// dealii::FEEvaluation::reinit() with #cell_ids suffices to read their DoF values.
      bool all_held;
    };

    /**
     * Set up the subcell index data and the face neighbors for the DoFHandler @p dof_no and the
     * quadrature @p quad_no of @p matrix_free. The DoF values of the extra cells are not exchanged,
     * call update_ghost_values() for that.
     */
    void
    reinit(const dealii::MatrixFree<dim, number> &matrix_free,
           const unsigned int                     dof_no,
           const unsigned int                     quad_no);

    /**
     * Exchange the DoF values of the extra cells from the locally owned values of @p src. The
     * ghost values of @p src are neither needed nor changed. The exchange is blocking.
     */
    void
    update_ghost_values(const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

    /**
     * Return whether the DoF values of the extra cells were last exchanged from @p src. Only the
     * address of @p src is compared, so changes of its values after update_ghost_values() are not
     * detected. Intended for assertions in the subcell evaluators.
     */
    bool
    has_ghost_values_of(const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

    /**
     * Return the neighbors across the face @p face_no of the cells in the batch @p cell_batch.
     * Both locally owned and ghost cell batches are supported. Intended for the cell loop.
     */
    const FaceNeighbors &
    get_face_neighbors(const unsigned int cell_batch, const unsigned int face_no) const;

    /**
     * Return the neighbors across the face @p face_no of the cells given by their dealii::MatrixFree
     * cell indices @p cell_ids, e.g. the cells on one side of a face batch. Lanes with an invalid
     * cell index are skipped. Intended for the face loop.
     */
    FaceNeighbors
    get_face_neighbors(const std::array<unsigned int, n_lanes> &cell_ids,
                       const unsigned int                       face_no) const;

    /**
     * Write the DoF values of the extra cell @p extra_cell into the lane @p lane of @p dof_values,
     * in the order of dealii::FEEvaluation::begin_dof_values() for all components of the element.
     * Requires a previous call to update_ghost_values().
     */
    void
    read_extra_cell_dof_values(const unsigned int   extra_cell,
                               const unsigned int   lane,
                               VectorizedArrayType *dof_values) const;

    /**
     * Return the number of extra cells, i.e. the number of ghost cells that are face neighbors of
     * held ghost cells but not held by the dealii::MatrixFree object themselves.
     */
    unsigned int
    n_extra_cells() const;

    /**
     * Return the subcell index data that is the same for all cells.
     */
    const SubcellShapeInfo<dim, number> &
    get_subcell_shape_info() const;

    /**
     * Return the underlying dealii::MatrixFree object.
     */
    const dealii::MatrixFree<dim, number> &
    get_matrix_free() const;

    /**
     * Return the index of the DoFHandler within the dealii::MatrixFree object.
     */
    unsigned int
    get_dof_no() const;

    /**
     * Return the index of the quadrature within the dealii::MatrixFree object.
     */
    unsigned int
    get_quad_no() const;

  private:
    /**
     * Set up the face neighbors of all held cells and collect the extra cells.
     */
    void
    setup_face_neighbors();

    /**
     * Set up the partitioner and the DoF positions for the exchange of the DoF values of the extra
     * cells.
     */
    void
    setup_extra_cells();

    /// The underlying matrix-free object.
    const dealii::MatrixFree<dim, number> *matrix_free = nullptr;

    /// Index of the DoFHandler within the matrix-free object.
    unsigned int dof_no = dealii::numbers::invalid_unsigned_int;

    /// Index of the quadrature within the matrix-free object.
    unsigned int quad_no = dealii::numbers::invalid_unsigned_int;

    /// Number of DoFs per cell, including all components.
    unsigned int n_dofs_per_cell = 0;

    /// Subcell index data that is the same for all cells.
    SubcellShapeInfo<dim, number> subcell_shape_info;

    /// [owned + ghost cell batch][face_no]: neighbors across the face.
    dealii::Table<2, FaceNeighbors> face_neighbors;

    /// [extra cell]: the extra cells as DoFHandler cell iterators.
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> extra_cells;

    /// Partitioner with the locally owned DoFs of the DoFHandler and the DoFs of the extra cells
    /// as ghost indices.
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> extra_cell_partitioner;

    /// DoF values of the extra cells in the order of the ghost indices of #extra_cell_partitioner.
    mutable dealii::AlignedVector<number> extra_cell_ghost_values;

    /// Temporary storage for the values sent to other processes in update_ghost_values().
    mutable dealii::AlignedVector<number> extra_cell_import_values;

    /// [extra cell * n_dofs_per_cell + i]: position of the DoF value in #extra_cell_ghost_values,
    /// with the DoFs of a cell in the order of dealii::FEEvaluation.
    std::vector<unsigned int> extra_cell_dof_positions;

    /// The vector from which the DoF values of the extra cells were last exchanged.
    mutable const dealii::LinearAlgebra::distributed::Vector<number> *ghost_values_source = nullptr;
  };
} // namespace MeltPoolDG
