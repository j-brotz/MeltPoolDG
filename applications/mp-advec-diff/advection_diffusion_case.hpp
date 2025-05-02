#pragma once
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  struct AdvectionDiffusionCaseParameters : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      amr.add_parameters(prm);
      advec_diff.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) final
    {
      amr.post(base.global_refinements, false /*restart not supported*/);
      advec_diff.post(base.fe);
      output.post(time_stepping.time_step_size, parameter_filename);

      // check input parameters for validity
      advec_diff.check_input_parameters();
      profiling.check_input_parameters(time_stepping.time_step_size);
    }

  public:
    BaseData                                  base;
    TimeIntegration::TimeSteppingData<number> time_stepping;
    AdaptiveMeshingData<number>               amr;
    AdvectionDiffusionData<number>            advec_diff;
    OutputData<number>                        output;
    Profiling::ProfilingData<number>          profiling;
  };

  template <int dim, typename number>
  class AdvectionDiffusionCase : public SimulationCaseBase<dim, number>
  {
  public:
    AdvectionDiffusionCaseParameters<number> parameters;

    AdvectionDiffusionCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG::LevelSet
