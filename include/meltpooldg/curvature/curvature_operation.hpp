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
#include <meltpooldg/normal_vector/normal_vector_operation.hpp>

namespace MeltPoolDG::Curvature
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
    /*
     *  In this struct, the main parameters of the curvature class are stored.
     */
    CurvatureData<double> curvature_data;

    CurvatureOperation() = default;

    void
    initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
               const Parameters<double> &                     data_in,
               unsigned int                                   curv_dof_idx_in,
               unsigned int                                   curv_quad_idx_in,
               unsigned int                                   normal_dof_idx_in,
               unsigned int                                   ls_dof_idx_in) override;

    void
    solve(const VectorType &solution_levelset) override;

    const LinearAlgebra::distributed::Vector<double> &
    get_curvature() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_curvature() override;

    const LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector() const override;


    void
    reinit();

  private:
    void
    create_operator();

    std::shared_ptr<const ScratchData<dim>> scratch_data;

    NormalVector::NormalVectorOperation<dim> normal_vector_operation;

    /*
     *  This pointer will point to your user-defined curvature operator.
     */
    std::unique_ptr<OperatorBase<double, VectorType, BlockVectorType>> curvature_operator;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
     *  multiple DoFHandlers, quadrature rules, etc.
     */
    unsigned int curv_dof_idx;
    unsigned int curv_quad_idx;
    unsigned int normal_dof_idx;
    /*
     *    This is the primary solution variable of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType solution_curvature;
  };
} // namespace MeltPoolDG::Curvature
