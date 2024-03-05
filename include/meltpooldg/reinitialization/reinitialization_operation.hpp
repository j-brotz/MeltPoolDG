/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// for using smart pointers
#include <deal.II/base/smartpointer.h>

// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation_adaflo_wrapper.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/reinitialization/olsson_operator.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    /*
     *     Reinitialization model for reobtaining the signed-distance
     *     property of the level set equation
     */

    template <int dim>
    class ReinitializationOperation : public ReinitializationOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      ReinitializationOperation(const ScratchData<dim>             &scratch_data_in,
                                const ReinitializationData<double> &reinit_data,
                                const NormalVectorData<double>     &normal_vec_data,
                                const int                           ls_n_subdivisions_in,
                                const TimeIterator<double>         &time_iterator,
                                const unsigned int                  reinit_dof_idx_in,
                                const unsigned int                  reinit_quad_idx_in,
                                const unsigned int                  ls_dof_idx_in,
                                const unsigned int                  normal_dof_idx_in);

      void
      reinit() override;

      /*
       *  By calling the reinitialize function, (1) the solution_level_set field
       *  and (2) the normal vector field corresponding to the given solution_level_set_field
       *  is updated. This is commonly the first stage before performing the pseudo-time-dependent
       *  solution procedure.
       */
      void
      set_initial_condition(const VectorType &solution_level_set_in) override;

      void
      update_dof_idx(const unsigned int &reinit_dof_idx_in) override;

      void
      init_time_advance();

      void
      solve() override;

      double
      get_max_change_level_set() const final;

      const BlockVectorType &
      get_normal_vector() const override;

      const VectorType &
      get_level_set() const override;

      VectorType &
      get_level_set() override;

      BlockVectorType &
      get_normal_vector() override;

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

      void
      attach_output_vectors(GenericDataOut<dim> &data_out) const override;


    private:
      void
      create_operator();

      void
      update_operator();

    private:
      const ScratchData<dim>            &scratch_data;
      const ReinitializationData<double> reinit_data;
      const int                          ls_n_subdivisions;
      const TimeIterator<double>        &time_iterator;
      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      mutable unsigned int reinit_dof_idx;
      const unsigned int   reinit_quad_idx;
      const unsigned int   ls_dof_idx;
      const unsigned int   normal_dof_idx;

      TimeIntegration::SolutionHistory<VectorType> solution_history;

      std::unique_ptr<Predictor<VectorType, double>> predictor;
      /*
       *  This shared pointer will point to your user-defined reinitialization operator.
       */
      std::unique_ptr<OperatorBase<dim, double>> reinit_operator;
      /*
       *   Computation of the normal vectors
       */
      std::shared_ptr<NormalVectorOperationBase<dim>> normal_vector_operation;
      /*
       *    This is the primary solution variable of this module, which will be also publically
       *    accessible for output_results.
       */
      VectorType solution_level_set;

      VectorType delta_psi_extrapolated;
      VectorType rhs;

      /*
       * Preconditioner for the matrix-free reinitialization operator
       */
      std::shared_ptr<
        Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>
        preconditioner_matrixfree;
      /*
       * Cache for diagonal preconditioner matrix-free
       */
      std::shared_ptr<DiagonalMatrix<VectorType>> diag_preconditioner_matrixfree;
      /*
       * Cache for trilinos preconditioner matrix-free
       */
      std::shared_ptr<TrilinosWrappers::PreconditionBase> trilinos_preconditioner_matrixfree;
      /*
       * Flag, if preconditioner matrices should be updated
       */
      bool update_preconditioner_matrixfree = true;

      // maximum change of the level set due to the current reinitialization step
      double max_change_level_set = std::numeric_limits<double>::max();
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
