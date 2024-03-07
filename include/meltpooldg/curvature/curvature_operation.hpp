/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// MeltPoolDG
#include <meltpooldg/curvature/curvature_operation_base.hpp>
#include <meltpooldg/curvature/curvature_operator.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  class CurvatureOperation : public CurvatureOperationBase<dim>
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
    using VectorType       = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
    using SparseMatrixType = TrilinosWrappers::SparseMatrix;

  public:
    CurvatureOperation(const ScratchData<dim>         &scratch_data_in,
                       const CurvatureData<double>    &curvature_data,
                       const NormalVectorData<double> &normal_vec_data,
                       const VectorType               &solution_levelset,
                       const unsigned int              curv_dof_idx_in,
                       const unsigned int              curv_quad_idx_in,
                       const unsigned int              normal_dof_idx_in,
                       const unsigned int              ls_dof_idx_in);

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
    reinit() override;

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) final;

  private:
    void
    create_operator(const VectorType &solution_levelset);

    const ScratchData<dim> &scratch_data;

    const CurvatureData<double> curvature_data;

    const VectorType &solution_levelset;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
     *  multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int curv_dof_idx;
    const unsigned int curv_quad_idx;
    const unsigned int normal_dof_idx;
    const unsigned int ls_dof_idx;

    LevelSet::NormalVectorOperation<dim> normal_vector_operation;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::unique_ptr<Predictor<VectorType, double>> predictor;

    /*
     *  This pointer will point to your user-defined curvature operator.
     */
    std::shared_ptr<CurvatureOperator<dim>> curvature_operator;
    /*
     *    This is the primary solution variable of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType solution_curvature_predictor;
    VectorType rhs;
    /*
     * Preconditioner for the matrix-free curvature operator
     */
    std::shared_ptr<Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>
      preconditioner_matrixfree;
    /*
     * Cache for diagonal preconditioner matrix-free
     */
    std::shared_ptr<DiagonalMatrix<VectorType>> diag_preconditioner_matrixfree;
    /*
     * Cache for trilinos preconditioner matrix-free
     */
    std::shared_ptr<TrilinosWrappers::PreconditionBase> trilinos_preconditioner_matrixfree;
  };
} // namespace MeltPoolDG::LevelSet
