#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <memory>
#include <string>

#include "reinitialization_case.hpp"


namespace MeltPoolDG
{
  namespace LevelSet
  {
    template <int dim, typename number>
    class ReinitializationApplication
    {
    private:
      using CaseType       = ReinitializationCase<dim, number>;
      using VectorType     = dealii::LinearAlgebra::distributed::Vector<number>;
      using DoFHandlerType = dealii::DoFHandler<dim>;

    public:
      ReinitializationApplication(std::unique_ptr<CaseType> simulation_case_)
        : simulation_case(std::move(simulation_case_))
        , param(simulation_case->parameters)
      {}

      void
      run();

    private:
      std::unique_ptr<CaseType>                     simulation_case;
      const ReinitializationCaseParameters<number> &param;
      DoFHandlerType                                dof_handler;
      dealii::AffineConstraints<number>             constraints;

      std::shared_ptr<ScratchData<dim, dim, number>>              scratch_data;
      std::unique_ptr<TimeIntegration::TimeIterator<number>>      time_iterator;
      std::unique_ptr<ReinitializationOperationBase<dim, number>> reinit_operation;
      unsigned int                                                reinit_dof_idx  = -1;
      unsigned int                                                normal_dof_idx  = -1;
      unsigned int                                                reinit_quad_idx = -1;

      std::unique_ptr<Postprocessor<dim, number>> post_processor;

      void
      initialize();

      void
      setup_dof_system();

      void
      refine_mesh();

      void
      output_results(const unsigned int time_step, const number current_time);
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
