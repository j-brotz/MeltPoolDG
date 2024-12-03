#pragma once
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

#include <string>


namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <typename number = double>
  struct CompressibleFlowCaseParameters final : public ParametersBase
  {
  protected:
    void
    add_parameters(ParameterHandler &prm) override
    {
      base.add_parameters(prm);
      flow.add_parameters(prm);
      time_integrator.add_parameters(prm);
      time_stepping.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) override
    {
      output.post(time_stepping.time_step_size, parameter_filename);
      profiling.post(base.verbosity_level);

      // check input parameters for validity
      profiling.check_input_parameters(time_stepping.time_step_size);
      base.fe.type = FiniteElementType::FE_DGQ; // only discontinuous elements are supported
    }

  public:
    BaseData                            base;
    CompressibleFlowData                flow;
    TimeIntegration::TimeIntegratorData time_integrator;
    TimeSteppingData<number>            time_stepping;
    OutputData<number>                  output;
    Profiling::ProfilingData<number>    profiling;
  };

  template <int dim>
  class CompressibleFlowCase : public SimulationCaseBase<dim>
  {
  public:
    CompressibleFlowCaseParameters<double> parameters;

    CompressibleFlowCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim>(parameter_file_in, mpi_communicator_in)
    {
      ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }

  protected:
    /**
     * This function calculates and prints the l2-norm of the solution given in @p generic_data_out
     * compared to a reference state given by the function @p reference_function, i.e. it computes
     * ||solution-reference||_2 for the primary variables density, momentum and energy density. The
     * string @p norm_name is printed to the console to indicate what the calculated values
     * represent.
     */
    void
    print_relative_norm(const GenericDataOut<dim> &generic_data_out,
                        dealii::Function<dim>     &reference_function,
                        const std::string         &norm_name = "Norm") const
    {
      const dealii::ConditionalOStream pcout(
        std::cout, Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);

      reference_function.set_time(generic_data_out.get_time());

      double      errors_squared[3] = {};
      const auto &dof_handler       = generic_data_out.get_dof_handler("density");
      const auto &dof_vector        = generic_data_out.get_vector("density");

      std::map<types::global_dof_index, Point<dim>> support_points =
        DoFTools::map_dofs_to_support_points(generic_data_out.get_mapping(), dof_handler);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_artificial() || cell->is_ghost())
            continue;
          const unsigned int nodes_per_cell = cell->get_fe().dofs_per_cell / (dim + 2);
          std::vector<types::global_dof_index> local_dof_indices(cell->get_fe().dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
            {
              switch (unsigned int component = i / nodes_per_cell)
                {
                    case 0: {
                      const double error =
                        dof_vector[local_dof_indices[i]] -
                        reference_function.value(support_points[local_dof_indices[i]], 0);
                      errors_squared[0] += error * error;
                      break;
                    }
                    case dim + 1: {
                      const double error =
                        dof_vector[local_dof_indices[i]] -
                        reference_function.value(support_points[local_dof_indices[i]], dim + 1);
                      errors_squared[2] += error * error;
                      break;
                    }
                    default: {
                      Assert(component > 0 && component < dim + 1, dealii::ExcInternalError());
                      const double error =
                        dof_vector[local_dof_indices[i]] -
                        reference_function.value(support_points[local_dof_indices[i]], component);
                      errors_squared[1] += error * error;
                    }
                }
            }
        }
      Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);
      std::array<double, 3> errors;
      for (unsigned int d = 0; d < 3; ++d)
        errors[d] = std::sqrt(errors_squared[d]);



      std::ostringstream output;
      output << norm_name << " rho: " << std::setprecision(4) << errors[0]
             << ", rho * u: " << std::setprecision(4) << errors[1]
             << ", rho * energy: " << std::setprecision(4) << errors[2];
      Journal::print_line(pcout, output.str(), "compressible_flow");
    }
  };
} // namespace MeltPoolDG::Flow