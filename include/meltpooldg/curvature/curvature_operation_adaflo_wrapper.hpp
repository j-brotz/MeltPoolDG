/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, December 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

// for parallelization
#  include <deal.II/lac/generic_linear_algebra.h>

// MeltPoolDG
#  include <meltpooldg/curvature/curvature_operation_base.hpp>
#  include <meltpooldg/interface/operator_base.hpp>
#  include <meltpooldg/interface/parameters.hpp>
#  include <meltpooldg/normal_vector/normal_vector_operation_adaflo_wrapper.hpp>

#  include <adaflo/level_set_okz_compute_curvature.h>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  class CurvatureOperationAdaflo : public CurvatureOperationBase<dim>
  {
  private:
    using VectorType       = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
    using SparseMatrixType = TrilinosWrappers::SparseMatrix;

  public:
    /**
     * Constructor.
     */
    CurvatureOperationAdaflo(const ScratchData<dim>   &scratch_data,
                             const int                 advec_diff_dof_idx,
                             const int                 normal_vec_dof_idx,
                             const int                 curv_dof_idx,
                             const int                 curv_quad_idx,
                             const VectorType         &advected_field,
                             const Parameters<double> &data_in);

    void
    reinit() override;

    /**
     * Solver time step
     */
    void
    solve() override;

    void
    update_normal_vector() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_curvature() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_curvature() override;

    const LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector() const override;

    LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector() override;

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

  private:
    void
    create_operator();

    void
    create_normal_vector_operator();

    void
    set_adaflo_parameters(const Parameters<double> &parameters,
                          int                       advec_diff_dof_idx,
                          int                       curv_dof_idx,
                          int                       curv_quad_idx);

    void
    initialize_vectors();

    const ScratchData<dim> &scratch_data;
    const VectorType       &advected_field;
    /**
     *  Vectors for computing the normals
     */
    BlockVectorType normal_vec_dummy;
    VectorType      curvature_field;
    VectorType      rhs;
    /**
     * Adaflo parameters for the curvature problem
     */
    LevelSetOKZSolverComputeCurvatureParameter curv_adaflo_params;
    /**
     * Reference to the actual curvature solver from adaflo
     */
    std::shared_ptr<LevelSetOKZSolverComputeCurvature<dim>>     curvature_operation;
    std::shared_ptr<LevelSet::NormalVectorOperationAdaflo<dim>> normal_vector_operation_adaflo;

    /**
     *  Diagonal preconditioner
     */
    DiagonalPreconditioner<double> preconditioner;
    /**
     *  Projection matrices
     */
    std::shared_ptr<BlockMatrixExtension> projection_matrix;
    std::shared_ptr<BlockILUExtension>    ilu_projection_matrix;
    /**
     *  Geometry info
     */
    AlignedVector<VectorizedArray<double>> cell_diameters;
    double                                 cell_diameter_min;
    double                                 cell_diameter_max;
    double                                 epsilon_used;
    unsigned int                           verbosity_level;

    const NormalVectorData<double> &normal_vector_data;
  };
} // namespace MeltPoolDG::LevelSet

#endif
