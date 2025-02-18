#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>


namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <typename number = double>
  struct ReinitializationCaseParameters : public ParametersBase
  {
  protected:
    void
    add_parameters(ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      amr.add_parameters(prm);
      reinit.add_parameters(prm);
      normal_vec.add_parameters(prm);
      curv.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) final
    {
      amr.post(base.global_refinements, false /*restart not supported*/);
      reinit.post(base.fe);
      normal_vec.post(base.verbosity_level);
      curv.post(base.verbosity_level);
      output.post(time_stepping.time_step_size, parameter_filename);

      // check input parameters for validity
      reinit.check_input_parameters(normal_vec.linear_solver.do_matrix_free);
      normal_vec.check_input_parameters(reinit.interface_thickness_parameter.type);
      curv.check_input_parameters(reinit.interface_thickness_parameter.type);
      profiling.check_input_parameters(time_stepping.time_step_size);
    }

  public:
    BaseData                         base;
    TimeSteppingData<number>         time_stepping;
    AdaptiveMeshingData              amr;
    ReinitializationData<number>     reinit;
    NormalVectorData<number>         normal_vec;
    CurvatureData<number>            curv;
    OutputData<number>               output;
    Profiling::ProfilingData<number> profiling;
  };

  template <int dim>
  class ReinitializationCase : public SimulationCaseBase<dim>
  {
  public:
    ReinitializationCaseParameters<double> parameters;

    ReinitializationCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG::LevelSet
