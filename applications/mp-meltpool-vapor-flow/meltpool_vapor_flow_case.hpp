#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>

#include <string>

namespace MeltPoolDG::MeltPoolVaporFlow
{
  /**
   * @brief Struct that manages all relevant parameters for compressible flow simulations.
   */
  template <typename number>
  struct CaseParameters final : public ParametersBase
  {
  protected:
    /**
     * @brief Add all relevant parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm) override
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    /**
     * @brief Post-process parameters.
     *
     * @param parameter_filename Name of the parameter file.
     */
    void
    post(const std::string &parameter_filename) override final
    {
      output.post(time_stepping.time_step_size, parameter_filename);
      profiling.check_input_parameters(time_stepping.time_step_size);
    }

  public:
    /// Simulation basic data
    BaseData base;

    /// Data for time stepping
    TimeIntegration::TimeSteppingData<number> time_stepping;

    /// Data for output
    OutputData<number> output;

    /// Data for profiling
    Profiling::ProfilingData<number> profiling;
  };

  /**
   * @brief Case base class for compressible flow cases.
   */
  template <int dim, typename number>
  class Case : public SimulationCaseBase<dim, number>
  {
  public:
    /// Case-specific parameters
    CaseParameters<number> parameters;

    /**
     * @brief Constructor.
     *
     * @param parameter_file_in Parameter file that contains simulation input settings.
     * @param mpi_communicator_in The MPI communicator used to run the simulation in parallel.
     */
    explicit Case(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG::MeltPoolVaporFlow
