/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>

// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation_base.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  namespace NormalVector
  {
    using namespace dealii;

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

    template <int dim>
    class NormalVectorOperation : public NormalVectorOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      NormalVectorData<double> normal_vector_data;
      /*
       *    This is the primary solution variable of this module, which will be also publically
       *    accessible for output_results.
       */
      BlockVectorType solution_normal_vector;

      NormalVectorOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             normal_dof_idx_in,
                 const unsigned int                             normal_quad_idx_in,
                 const unsigned int                             ls_dof_idx_in) override;

      void
      reinit() override;

      void
      solve(const VectorType &solution_levelset_in) override;

      const BlockVectorType &
      get_solution_normal_vector() const override;

      BlockVectorType &
      get_solution_normal_vector() override;

    private:
      /**
       * This function creates the normal vector operator for assembling the system operator (either
       * matrix based or matrix free) and the right handside.
       */
      void
      create_operator();

    private:
      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  This pointer will point to your user-defined normal vector operator.
       */
      std::unique_ptr<OperatorBase<double, BlockVectorType, VectorType>> normal_vector_operator;
      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      unsigned int normal_dof_idx;
      unsigned int normal_quad_idx;
      unsigned int ls_dof_idx;
    };
  } // namespace NormalVector
} // namespace MeltPoolDG
