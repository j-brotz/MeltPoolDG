#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>

// MeltPoolDG
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/level_set/normal_vector_operation_base.hpp>
#include <meltpooldg/level_set/normal_vector_operator.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  /**
   *  This function calculates the normal vector of the current level set function being
   *  the solution of an intermediate projection step
   *
   *              (w, n_ϕ)  + η_n (∇w, ∇n_ϕ)  = (w,∇ϕ)
   *                      Ω                 Ω            Ω
   *
   *  with test function w, the normal vector n_ϕ, damping parameter η_n and the
   *  level set function ϕ.
   *
   *    !!!!
   *          the normal vector field is not normalized to length one,
   *          it actually represents the gradient of the level set
   *          function
   *    !!!!
   */

  template <int dim, typename number>
  class NormalVectorOperation : public NormalVectorOperationBase<dim, number>
  {
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

  public:
    NormalVectorOperation(const ScratchData<dim, dim, number> &scratch_data_in,
                          const NormalVectorData<number>      &normal_vector_data,
                          const VectorType                    &solution_level_set,
                          const unsigned int                   normal_dof_idx_in,
                          const unsigned int                   normal_quad_idx_in,
                          const unsigned int                   ls_dof_idx_in);

    void
    reinit() override;

    void
    solve() override;

    const BlockVectorType &
    get_solution_normal_vector() const override;

    BlockVectorType &
    get_solution_normal_vector() override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

  private:
    /**
     * This function creates the normal vector operator for assembling the system operator (either
     * matrix based or matrix free) and the right handside.
     */
    void
    create_operator();

  private:
    const ScratchData<dim, dim, number> &scratch_data;
    const NormalVectorData<number>       normal_vector_data;
    const VectorType                    &solution_level_set;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int normal_dof_idx;
    const unsigned int normal_quad_idx;
    const unsigned int ls_dof_idx;

    TimeIntegration::SolutionHistory<BlockVectorType> solution_history;

    std::unique_ptr<Predictor<BlockVectorType, number>> predictor;
    /*
     *    This is the primary solution variable of this module, which will be also publically
     *    accessible for output_results.
     */
    BlockVectorType solution_normal_vector_predictor;
    BlockVectorType rhs;
    /*
     *  This pointer will point to your user-defined normal vector operator.
     */
    std::unique_ptr<NormalVectorOperator<dim, number>> normal_vector_operator;
    /*
     * Preconditioner for the curvature operator
     */
    Preconditioner<dim, BlockVectorType> preconditioner;
  };
} // namespace MeltPoolDG::LevelSet
