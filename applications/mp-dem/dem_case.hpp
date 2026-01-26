#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_material_data.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
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
  template <typename number>
  struct DemCaseParameters final : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) override
    {
      base.add_parameters(prm);
      obstacle_contact.add_parameters(prm);
      time_stepping.add_parameters(prm);
      obstacle_data.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
      restart.add_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) override
    {
      output.post(time_stepping.time_step_size, parameter_filename);
      restart.post(output.directory);

      // check input parameters for validity
      profiling.check_input_parameters(time_stepping.time_step_size);
      restart.check_input_parameters(time_stepping.time_step_size);
    }

  public:
    BaseData                                   base;
    SphericalParticleContactData<number>       obstacle_contact;
    TimeIntegration::TimeSteppingData<number>  time_stepping;
    ObstacleData<number>                       obstacle_data;
    OutputData<number>                         output;
    Profiling::ProfilingData<number>           profiling;
    Restart::RestartData<number>               restart;
  };

  template <int dim, typename number>
  class DemCase : public SimulationCaseBase<dim, number>
  {
  public:
    DemCaseParameters<number> parameters;

    DemCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG
