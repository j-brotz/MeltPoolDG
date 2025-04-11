#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/base/aligned_vector.h>
#  include <deal.II/base/vectorization.h>

#  include <deal.II/lac/la_parallel_block_vector.h>
#  include <deal.II/lac/la_parallel_vector.h>
#  include <deal.II/lac/trilinos_sparse_matrix.h>

#  include <meltpooldg/core/scratch_data.hpp>
#  include <meltpooldg/level_set/normal_vector_data.hpp>
#  include <meltpooldg/level_set/normal_vector_operation_base.hpp>

#  include <adaflo/block_matrix_extension.h>
#  include <adaflo/diagonal_preconditioner.h>
#  include <adaflo/level_set_okz_compute_normal.h>

#  include <memory>
#  include <vector>


namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class NormalVectorOperationAdaflo : public NormalVectorOperationBase<dim, number>
  {
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

  public:
    /**
     * Constructor.
     */
    NormalVectorOperationAdaflo(const ScratchData<dim, dim, number> &scratch_data,
                                const NormalVectorData<number>      &normal_vec_data_in,
                                const int                            advec_diff_dof_idx,
                                const int                            normal_vec_dof_idx,
                                const int                            normal_vec_quad_idx,
                                const VectorType                    &advected_field_in,
                                const number                         reinit_scale_factor_epsilon);

    void
    reinit() override;

    /**
     * Solver time step
     */
    void
    solve() override;

    /**
     *  getter
     */
    const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_solution_normal_vector() const override;

    dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_solution_normal_vector() override;

    adaflo::LevelSetOKZSolverComputeNormal<dim> &
    get_adaflo_obj();

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

  private:
    void
    set_adaflo_parameters(const number epsilon,
                          const int    advec_diff_dof_idx,
                          const int    normal_vec_dof_idx,
                          const int    normal_vec_quad_idx);

    void
    initialize_vectors();

  private:
    void
    create_operator();

    const ScratchData<dim, dim, number> &scratch_data;

    const VectorType &advected_field;

    /**
     *  Vectors for computing the normals
     */
    BlockVectorType normal_vector_field;
    BlockVectorType rhs;
    /**
     * Adaflo parameters for the level set problem
     */
    adaflo::LevelSetOKZSolverComputeNormalParameter normal_vec_adaflo_params;
    /**
     * Reference to the actual advection diffusion solver from adaflo
     */
    std::shared_ptr<adaflo::LevelSetOKZSolverComputeNormal<dim>> normal_vec_operation;

    /**
     *  Diagonal preconditioner @todo
     */
    adaflo::DiagonalPreconditioner<number>                 preconditioner;
    std::shared_ptr<adaflo::BlockMatrixExtension>          projection_matrix;
    std::shared_ptr<adaflo::BlockILUExtension>             ilu_projection_matrix;
    dealii::AlignedVector<dealii::VectorizedArray<number>> cell_diameters;
    number                                                 cell_diameter_min;
    number                                                 cell_diameter_max;
    number                                                 epsilon_used;

    const NormalVectorData<number> &normal_vec_data;
  };
} // namespace MeltPoolDG::LevelSet
#endif
