#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#include <meltpooldg/particles/cohesive_forces.hpp>
#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/amr_data.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/restart.hpp>

namespace MeltPoolDG
{
  /// Enum for the type of amr strategy used by the application.
  BETTER_ENUM(AMRStrategy, char, indicator, obstacle_surface);

  template <typename number>
  struct CfdDemCaseParameters final : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) override
    {
      amr.add_parameters(prm);
      base.add_parameters(prm);
      fluid_structure_interaction_data.add_parameters(prm);
      flow.add_parameters(prm);
      material.add_parameters(prm, true /*is_gas*/);
      time_stepping.add_parameters(prm);
      obstacle_data.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
      restart.add_parameters(prm);
      add_application_specific_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) override
    {
      amr.post(base.global_refinements, false);
      output.post(time_stepping.time_step_size, parameter_filename);
      flow.post(base.fe, base.verbosity_level);
      material.post(true /*is_gas*/);
      restart.post(output.directory);

      // check input parameters for validity
      profiling.check_input_parameters(time_stepping.time_step_size);
      restart.check_input_parameters(time_stepping.time_step_size);
    }

    void
    add_application_specific_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("application");
      {
        prm.add_parameter(
          "amr strategies",
          application.amr_strategies,
          "List of comma separated AMR strategies used to determine whether cells should be refined or coarsened. "
          "This parameter is only relevant when AMR is enabled.");
      }
      prm.leave_subsection();
    }

  public:
    struct
    {
      /// List of AMR strategies used to determine whether cells should be refined or coarsened.
      std::vector<AMRStrategy> amr_strategies;
    } application;

    AdaptiveMeshingData<number>                 amr;
    BaseData                                    base;
    FluidStructureInteractionData<number>       fluid_structure_interaction_data;
    CompressibleFlow::OperationData<number>     flow;
    CompressibleFlow::MaterialPhaseData<number> material;
    TimeIntegration::TimeSteppingData<number>   time_stepping;
    ObstacleData<number>                        obstacle_data;
    OutputData<number>                          output;
    Profiling::ProfilingData<number>            profiling;
    Restart::RestartData<number>                restart;
  };

  template <int dim, typename number>
  class CfdDemCase : public SimulationCaseBase<dim, number>
  {
  public:
    CfdDemCaseParameters<number> parameters;

    CfdDemCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }

  protected:
    /**
     * This function calculates and prints the l2-norm/l2-error of the solution given in
     * @p generic_data_out compared to a reference state given by the function
     * @p reference_function, i.e., it computes ||solution-reference||_2 for the primary variables
     * density, momentum and energy density. The string @p norm_name is printed to the console to
     * indicate what the calculated values represent. If the reference solution represents an exact
     * analytical solution, the norm_name @p "error" should be chosen. Otherwise, the @p norm_name
     * "norm" is appropriate for the l2-norm of the deviation to a certain defined reference state,
     * e.g. initial conditions. It is necessary to provide a reference function for both cases.
     *
     * @param generic_data_out The postprocessing data.
     * @param reference_function Analytical reference function for error/norm computation.
     * @param norm_name Choose between "norm" and "error".
     */
    void
    print_relative_norm(const GenericDataOut<dim, number> &generic_data_out,
                        dealii::Function<dim>             &reference_function,
                        const std::string                 &norm_name = "norm") const
    {
      const dealii::ConditionalOStream pcout(std::cout,
                                             dealii::Utilities::MPI::this_mpi_process(
                                               this->mpi_communicator) == 0 and
                                               parameters.base.verbosity_level >= 1);

      const auto &dof_vector  = generic_data_out.get_vector("density");
      const auto &dof_handler = generic_data_out.get_dof_handler("density");

      reference_function.set_time(generic_data_out.get_time());

      number errors_squared[3] = {};

      std::map<dealii::types::global_dof_index, dealii::Point<dim, number>> support_points =
        dealii::DoFTools::map_dofs_to_support_points(generic_data_out.get_mapping(), dof_handler);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_artificial() or cell->is_ghost())
            continue;
          const unsigned int nodes_per_cell = cell->get_fe().dofs_per_cell / (dim + 2);
          std::vector<dealii::types::global_dof_index> local_dof_indices(
            cell->get_fe().dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
            {
              switch (unsigned int component = i / nodes_per_cell)
                {
                    case 0: {
                      const number error =
                        dof_vector[local_dof_indices[i]] -
                        reference_function.value(support_points[local_dof_indices[i]], 0);
                      errors_squared[0] += error * error;
                      break;
                    }
                    case dim + 1: {
                      const number error =
                        dof_vector[local_dof_indices[i]] -
                        reference_function.value(support_points[local_dof_indices[i]], dim + 1);
                      errors_squared[2] += error * error;
                      break;
                    }
                    default: {
                      Assert(component > 0 and component < dim + 1, dealii::ExcInternalError());
                      const number error =
                        dof_vector[local_dof_indices[i]] -
                        reference_function.value(support_points[local_dof_indices[i]], component);
                      errors_squared[1] += error * error;
                    }
                }
            }
        }

      dealii::Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);
      std::array<number, 3> errors;
      for (unsigned int d = 0; d < 3; ++d)
        errors[d] = std::sqrt(errors_squared[d]);

      std::ostringstream output;
      output << std::scientific << std::setprecision(4) << norm_name << " rho: " << errors[0]
             << ", rho * u: " << errors[1] << ", rho * energy: " << errors[2];
      Journal::print_line(pcout, output.str(), "compressible_flow");
    }
  };
} // namespace MeltPoolDG
