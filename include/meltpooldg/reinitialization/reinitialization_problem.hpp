/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// for index set
#include <deal.II/base/index_set.h>
// for distributed triangulation
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria_base.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/error_estimator.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
// for mapping
#include <deal.II/fe/mapping.h>
// for simplex
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
// MeltPoolDG
#include <meltpooldg/interface/problem_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_adaflo_wrapper.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/postprocessor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>
// C++
#include <memory>
namespace MeltPoolDG
{
  namespace Reinitialization
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
  } // namespace Reinitialization
} // namespace MeltPoolDG
