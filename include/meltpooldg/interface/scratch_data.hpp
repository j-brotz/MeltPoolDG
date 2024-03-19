/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/

#pragma once
// for parallelization
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
// for mapping
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_tools.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>

#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>


namespace MeltPoolDG
{
  /**
   * Container containing mapping-, finite-element-, and quadrature-related
   * objects to be used either in matrix-based or in matrix-free context.
   */
  using namespace dealii;

  template <int dim,
            int spacedim                 = dim,
            typename number              = double,
            typename VectorizedArrayType = VectorizedArray<number>>
  class ScratchData
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<number>;

  public:
    ScratchData(const MPI_Comm     mpi_communicator,
                const unsigned int max_verbosity_level,
                const bool         do_matrix_free);

    /**
     * Setup everything in one go.
     */
    void
    reinit(const Mapping<dim, spacedim>                         &mapping,
           const std::vector<const DoFHandler<dim, spacedim> *> &dof_handler,
           const std::vector<const AffineConstraints<number> *> &constraint,
           const std::vector<Quadrature<dim>>                   &quad,
           const bool                                            enable_boundary_face_loops,
           const bool                                            enable_inner_face_loops);
    /**
     * Fill internal data structures step-by-step.
     */
    void
    set_mapping(const Mapping<dim, spacedim> &mapping);

    void
    set_mapping(const std::shared_ptr<Mapping<dim, spacedim>> mapping);

    unsigned int
    attach_dof_handler(const DoFHandler<dim, spacedim> &dof_handler);

    unsigned int
    attach_constraint_matrix(const AffineConstraints<number> &constraint);

    unsigned int
    attach_quadrature(const Quadrature<dim> &quadrature);

    void
    create_partitioning();

    void
    build(const bool enable_boundary_face_loops, const bool enable_inner_face_loops);

    /**
     * initialize vectors
     */
    void
    initialize_dof_vector(VectorType &vec, const unsigned int dof_idx) const;

    void
    initialize_dof_vector(BlockVectorType &vec, const unsigned int dof_idx) const;

    void
    initialize_bc_vector(VectorType &vec, const unsigned int dof_idx) const;

    void
    initialize_bc_vector(BlockVectorType &vec, const unsigned int dof_idx) const;
    /*
     * clear all member variables
     */
    void
    clear();

    /*
     * create a map of RemotePointEvaluation objects corresponding to a DoFHandler
     * @p dof_idx
     */
    void
    create_remote_point_evaluation(const unsigned int                        dof_idx,
                                   const std::function<std::vector<bool>()> &marked_vertices = {});

    /**
     * Getter functions.
     */
    const Mapping<dim, spacedim> &
    get_mapping() const;

    const FiniteElement<dim, spacedim> &
    get_fe(const unsigned int fe_index) const;

    const AffineConstraints<number> &
    get_constraint(const unsigned int constraint_index) const;

    AffineConstraints<number> &
    get_constraint(const unsigned int constraint_index);

    const std::vector<const AffineConstraints<number> *> &
    get_constraints() const;

    const Quadrature<dim> &
    get_quadrature(const unsigned int quad_index) const;

    const std::vector<Quadrature<dim>> &
    get_quadratures() const;

    const Quadrature<dim - 1> &
    get_face_quadrature(const unsigned int quad_index) const;

    const std::vector<Quadrature<dim - 1>> &
    get_face_quadratures() const;

    MatrixFree<dim, number, VectorizedArrayType> &
    get_matrix_free();

    const MatrixFree<dim, number, VectorizedArrayType> &
    get_matrix_free() const;

    const DoFHandler<dim, spacedim> &
    get_dof_handler(const unsigned int dof_idx) const;

    const std::vector<const DoFHandler<dim, spacedim> *> &
    get_dof_handlers() const;

    const Triangulation<dim> &
    get_triangulation(const unsigned int dof_idx = 0) const;

    unsigned int
    get_n_dofs_per_cell(const unsigned int dof_idx) const;

    unsigned int
    get_degree(const unsigned int dof_idx) const;

    unsigned int
    get_n_q_points(const unsigned int dof_idx) const;

    const double &
    get_min_cell_size() const;

    double
    get_min_cell_size(const unsigned int dof_idx) const;

    const double &
    get_max_cell_size() const;

    const double &
    get_min_diameter() const;

    const AlignedVector<VectorizedArray<double>> &
    get_cell_sizes() const;

    MPI_Comm
    get_mpi_comm(const unsigned int dof_idx = 0) const;

    const IndexSet &
    get_locally_owned_dofs(const unsigned int dof_idx) const;

    const IndexSet &
    get_locally_relevant_dofs(const unsigned int dof_idx) const;

    const std::shared_ptr<Utilities::MPI::Partitioner> &
    get_partitioner(const unsigned int dof_idx) const;

    const ConditionalOStream
    get_pcout(const unsigned int level = 0) const;

    bool
    is_hex_mesh(const unsigned int dof_idx = 0) const;

    bool
    is_FE_Q_iso_Q_1(const unsigned int dof_idx) const;

    TimerOutput &
    get_timer() const;

    Utilities::MPI::RemotePointEvaluation<dim, dim> &
    get_remote_point_evaluation(const unsigned int dof_idx) const;

    bool enable_boundary_faces;
    bool enable_inner_faces;

  private:
    bool                                           do_matrix_free;
    std::vector<dealii::ConditionalOStream>        pcout;
    std::shared_ptr<Mapping<dim, spacedim>>        mapping;
    std::vector<const DoFHandler<dim, spacedim> *> dof_handler;
    std::vector<const AffineConstraints<number> *> constraint;
    std::vector<Quadrature<dim>>                   quad;
    std::vector<Quadrature<dim - 1>>               face_quad;
    double                                         min_cell_size;
    double                                 max_cell_size; // only computed in case of matrixfree
    double                                 min_diameter;
    AlignedVector<VectorizedArray<double>> cell_sizes;
    std::vector<IndexSet>                  locally_owned_dofs;
    std::vector<IndexSet>                  locally_relevant_dofs;
    std::vector<std::shared_ptr<Utilities::MPI::Partitioner>> partitioner;
    std::map<unsigned int, std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim, dim>>> rpe;

    mutable std::shared_ptr<TimerOutput> timer;

    MatrixFree<dim, number, VectorizedArrayType> matrix_free;
  };
} // namespace MeltPoolDG
