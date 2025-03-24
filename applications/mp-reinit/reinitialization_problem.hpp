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

#include <meltpooldg/core/problem_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <memory>
#include <string>

#include "reinitialization_case.hpp"


namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    template <int dim>
    class ReinitializationProblem
    {
    private:
      using CaseType       = ReinitializationCase<dim>;
      using VectorType     = LinearAlgebra::distributed::Vector<double>;
      using DoFHandlerType = DoFHandler<dim>;

    public:
      ReinitializationProblem(std::unique_ptr<CaseType> simulation_case_)
        : simulation_case(std::move(simulation_case_))
        , param(simulation_case->parameters)
      {}

      void
      run();

    private:
      std::unique_ptr<CaseType>                     simulation_case;
      const ReinitializationCaseParameters<double> &param;
      DoFHandler<dim>                               dof_handler;
      AffineConstraints<double>                     constraints;

      std::shared_ptr<ScratchData<dim>>                   scratch_data;
      std::unique_ptr<TimeIterator<double>>               time_iterator;
      std::unique_ptr<ReinitializationOperationBase<dim>> reinit_operation;
      unsigned int                                        reinit_dof_idx  = -1;
      unsigned int                                        normal_dof_idx  = -1;
      unsigned int                                        reinit_quad_idx = -1;

      std::unique_ptr<Postprocessor<dim, double>> post_processor;

      void
      initialize();

      void
      setup_dof_system();

      void
      refine_mesh();

      void
      output_results(const unsigned int time_step, const double current_time);
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
