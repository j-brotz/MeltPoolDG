/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/interface/problem_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>
#include <string>


namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    /*
     *     Reinitialization model for reobtaining the "signed-distance"
     *     property of the level set equation
     */

    template <int dim>
    class ReinitializationProblem : public ProblemBase<dim>
    {
    private:
      using VectorType     = LinearAlgebra::distributed::Vector<double>;
      using DoFHandlerType = DoFHandler<dim>;

    public:
      /*
       *  Constructor of reinitialization problem
       */

      ReinitializationProblem() = default;

      void
      run(std::shared_ptr<SimulationBase<dim>> base_in) final;

      std::string
      get_name() final;

    private:
      /*
       *  This function initials the relevant member data
       *  for the computation of a reinitialization problem
       */
      void
      initialize(std::shared_ptr<SimulationBase<dim>> base_in);

      void
      setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in);

      /*
       *  perform mesh refinement
       */
      void
      refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in);

      /*
       *  perform output of results
       */
      void
      output_results(const unsigned int                   time_step,
                     const double                         current_time,
                     std::shared_ptr<SimulationBase<dim>> base_in);

    private:
      DoFHandler<dim>           dof_handler;
      AffineConstraints<double> constraints;

      std::shared_ptr<ScratchData<dim>>                   scratch_data;
      std::shared_ptr<TimeIterator<double>>               time_iterator;
      std::shared_ptr<ReinitializationOperationBase<dim>> reinit_operation;
      unsigned int                                        reinit_dof_idx;
      unsigned int                                        normal_dof_idx;
      unsigned int                                        reinit_quad_idx;

      std::shared_ptr<Postprocessor<dim>> post_processor;
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
