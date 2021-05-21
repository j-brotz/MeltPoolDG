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
#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation_adaflo_wrapper.hpp>
#include <meltpooldg/reinitialization/olsson_operator.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG
{
  namespace Reinitialization
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
      ReinitializationData<double> reinit_data;
      /*
       *    This is the primary solution variable of this module, which will be also publically
       *    accessible for output_results.
       */
      VectorType solution_level_set;

      ReinitializationOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             reinit_dof_idx_in,
                 const unsigned int                             reinit_quad_idx_in,
                 const unsigned int                             normal_dof_idx_in) override;

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
      solve(const double d_tau) override;

      const BlockVectorType &
      get_normal_vector() const override;

      const VectorType &
      get_level_set() const override;

      VectorType &
      get_level_set() override;

      BlockVectorType &
      get_normal_vector() override;

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

      void
      attach_output_vectors(GenericDataOut<dim> &data_out) const;


    private:
      void
      set_reinitialization_parameters(const Parameters<double> &data_in);

      void
      create_operator();

      void
      update_operator();

    private:
      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  This shared pointer will point to your user-defined reinitialization operator.
       */
      std::unique_ptr<OperatorBase<dim, double>> reinit_operator;
      /*
       *   Computation of the normal vectors
       */
      std::shared_ptr<NormalVector::NormalVectorOperationBase<dim>> normal_vector_operation;
      // NormalVector::NormalVectorOperation<dim> normal_vector_operation;
      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      unsigned int reinit_dof_idx;
      unsigned int reinit_quad_idx;
      unsigned int normal_dof_idx;
    };
  } // namespace Reinitialization
} // namespace MeltPoolDG
