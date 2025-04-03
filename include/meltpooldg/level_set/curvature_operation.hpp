#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// MeltPoolDG
#include <meltpooldg/level_set/curvature_operation_base.hpp>
#include <meltpooldg/level_set/curvature_operator.hpp>
#include <meltpooldg/level_set/normal_vector_operation.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

namespace MeltPoolDG::LevelSet
{


  template <int dim, typename number>
  class CurvatureOperation : public CurvatureOperationBase<dim, number>
  {
    /*
     *  This function calculates the curvature of the current level set function being
     *  the solution of an intermediate projection step
     *
     *              (w, κ)   +   η_κ (∇w, ∇κ)  = (w,∇·n_ϕ)
     *                    Ω                  Ω            Ω
     *
     *  with test function w, curvature κ, damping parameter η_κ and the normal to the
     *  level set function n_ϕ.
     *
     */
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

  public:
    CurvatureOperation(const ScratchData<dim, dim, number> &scratch_data_in,
                       const CurvatureData<number>         &curvature_data,
                       const NormalVectorData<number>      &normal_vec_data,
                       const VectorType                    &solution_levelset,
                       const unsigned int                   curv_dof_idx_in,
                       const unsigned int                   curv_quad_idx_in,
                       const unsigned int                   normal_dof_idx_in,
                       const unsigned int                   ls_dof_idx_in);

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
    reinit() override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) final;

  private:
    void
    create_operator(const VectorType &solution_levelset);

    const ScratchData<dim, dim, number> &scratch_data;

    const CurvatureData<number> curvature_data;

    const VectorType &solution_levelset;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int curv_dof_idx;
    const unsigned int curv_quad_idx;
    const unsigned int normal_dof_idx;
    const unsigned int ls_dof_idx;

    LevelSet::NormalVectorOperation<dim, number> normal_vector_operation;

    TimeIntegration::SolutionHistory<VectorType, number> solution_history;

    std::unique_ptr<Predictor<VectorType, number>> predictor;

    /*
     *  This pointer will point to your user-defined curvature operator.
     */
    std::shared_ptr<CurvatureOperator<dim, number>> curvature_operator;
    /*
     *    This is the primary solution variable of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType solution_curvature_predictor;
    VectorType rhs;
    /*
     * Preconditioner for the matrix-free curvature operator
     */
    Preconditioner<dim, VectorType> preconditioner;
  };
} // namespace MeltPoolDG::LevelSet
