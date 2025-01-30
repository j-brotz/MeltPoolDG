#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/utilities/amr_data.hpp>
#include <meltpooldg/utilities/material_data.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

#include <string>


namespace MeltPoolDG::RadiativeTransport
{
  template <typename number = double>
  struct RadiativeTransportCaseParameters : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      amr.add_parameters(prm);
      rad_trans.add_parameters(prm);
      material.add_parameters(prm);
      heat.add_parameters(prm);
      laser.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) final
    {
      amr.post(base.global_refinements, false /*restart not supported*/);
      heat.post(base.fe, base.verbosity_level);
      laser.post(base.dimension,
                 heat.use_volume_specific_thermal_capacity_for_phase_interpolation,
                 material);
      output.post(time_stepping.time_step_size, parameter_filename);
      profiling.post(base.verbosity_level);

      base.check_input_parameters(1);
      heat.check_input_parameters(base.fe);
      laser.check_input_parameters();
      profiling.check_input_parameters(time_stepping.time_step_size);
    }

  public:
    BaseData                         base;
    TimeSteppingData<number>         time_stepping;
    AdaptiveMeshingData              amr;
    RadiativeTransportData<number>   rad_trans;
    MaterialData<number>             material;
    Heat::HeatData<number>           heat;
    Heat::LaserData<number>          laser;
    OutputData<number>               output;
    Profiling::ProfilingData<number> profiling;
  };

  template <int dim>
  class RadiativeTransportCase : public SimulationCaseBase<dim>
  {
  public:
    RadiativeTransportCaseParameters<double> parameters;

    RadiativeTransportCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG::RadiativeTransport
