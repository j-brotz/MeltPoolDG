#pragma once

#ifdef MPDG_ENABLE_ADAFLO

// for parallelization
#  include <deal.II/lac/generic_linear_algebra.h>

// MeltPoolDG
#  include <meltpooldg/core/operator_base.hpp>
#  include <meltpooldg/level_set/curvature_operation_base.hpp>
#  include <meltpooldg/level_set/level_set_data.hpp>
#  include <meltpooldg/level_set/normal_vector_operation_adaflo_wrapper.hpp>

#  include <adaflo/level_set_okz_compute_curvature.h>

namespace MeltPoolDG::LevelSet
{


  template <int dim, typename number>
  class CurvatureOperationAdaflo : public CurvatureOperationBase<dim, number>
  {
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

  public:
    /**
     * Constructor.
     */
    CurvatureOperationAdaflo(const ScratchData<dim, dim, number> &scratch_data,
                             const int                            advec_diff_dof_idx,
                             const int                            normal_vec_dof_idx,
                             const int                            curv_dof_idx,
                             const int                            curv_quad_idx,
                             const VectorType                    &advected_field,
                             const LevelSetData<number>          &data_in);

    void
    reinit() override;

    /**
     * Solver time step
     */
    void
    solve() override;

    void
    update_normal_vector() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() override;

    const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() const override;

    dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

  private:
    void
    create_operator();

    void
    create_normal_vector_operator();

    void
    set_adaflo_parameters(const LevelSetData<number> &parameters,
                          int                         advec_diff_dof_idx,
                          int                         curv_dof_idx,
                          int                         curv_quad_idx);

    void
    initialize_vectors();

    const ScratchData<dim, dim, number> &scratch_data;
    const VectorType                    &advected_field;
    /**
     *  Vectors for computing the normals
     */
    BlockVectorType normal_vec_dummy;
    VectorType      curvature_field;
    VectorType      rhs;
    /**
     * Adaflo parameters for the curvature problem
     */
    adaflo::LevelSetOKZSolverComputeCurvatureParameter curv_adaflo_params;
    /**
     * Reference to the actual curvature solver from adaflo
     */
    std::shared_ptr<adaflo::LevelSetOKZSolverComputeCurvature<dim>> curvature_operation;
    std::shared_ptr<LevelSet::NormalVectorOperationAdaflo<dim, number>>
      normal_vector_operation_adaflo;

    /**
     *  Diagonal preconditioner
     */
    adaflo::DiagonalPreconditioner<number> preconditioner;
    /**
     *  Projection matrices
     */
    std::shared_ptr<adaflo::BlockMatrixExtension> projection_matrix;
    std::shared_ptr<adaflo::BlockILUExtension>    ilu_projection_matrix;
    /**
     *  Geometry info
     */
    dealii::AlignedVector<dealii::VectorizedArray<number>> cell_diameters;
    number                                                 cell_diameter_min;
    number                                                 cell_diameter_max;
    number                                                 epsilon_used;
    unsigned int                                           verbosity_level;

    const NormalVectorData<number> &normal_vector_data;
  };
} // namespace MeltPoolDG::LevelSet

#endif
