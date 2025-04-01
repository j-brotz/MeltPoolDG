#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/utilities/amr_data.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/material_data.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

#include <string>


namespace MeltPoolDG::Heat
{
  BETTER_ENUM(AMRStrategy, char, KellyErrorEstimator, generic)

  template <typename number = double>
  struct HeatTransferCaseParameters : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      amr.add_parameters(prm);
      heat.add_parameters(prm);
      material.add_parameters(prm);
      laser.add_parameters(prm);
      rad_trans.add_parameters(prm);
      evapor.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);

      prm.enter_subsection("problem specific");
      {
        prm.add_parameter(
          "do solidification",
          problem_specific_parameters.do_solidification,
          "Set this parameter to true if you want to consider melting/solidification effects.");
        prm.add_parameter("amr strategy",
                          problem_specific_parameters.amr_strategy,
                          "Select the AMR strategy.");
      }
      prm.leave_subsection();
    }

    void
    post(const std::string &parameter_filename) final
    {
      amr.post(base.global_refinements, false /*restart not supported*/);
      heat.post(base.fe, base.verbosity_level);
      laser.post(base.dimension,
                 heat.diffuse.use_volume_specific_thermal_capacity_for_phase_interpolation,
                 material);
      rad_trans.post(base.fe, base.verbosity_level);
      evapor.post(material,
                  heat.diffuse.use_volume_specific_thermal_capacity_for_phase_interpolation);
      output.post(time_stepping.time_step_size, parameter_filename);

      base.check_input_parameters(1);
      heat.check_input_parameters(base.fe);
      laser.check_input_parameters();
      rad_trans.check_input_parameters(base.fe);
      evapor.check_input_parameters(material, 1);
      profiling.check_input_parameters(time_stepping.time_step_size);
    }

  public:
    BaseData                                           base;
    TimeSteppingData<number>                           time_stepping;
    AdaptiveMeshingData<number>                        amr;
    HeatData<number>                                   heat;
    MaterialData<number>                               material;
    LaserData<number>                                  laser;
    RadiativeTransport::RadiativeTransportData<number> rad_trans;
    Evaporation::EvaporationData<number>               evapor;
    OutputData<number>                                 output;
    Profiling::ProfilingData<number>                   profiling;

    struct
    {
      bool        do_solidification = false;
      AMRStrategy amr_strategy      = AMRStrategy::KellyErrorEstimator;
    } problem_specific_parameters;
  };


  template <int dim>
  class HeatTransferCase : public SimulationCaseBase<dim>
  {
  public:
    HeatTransferCaseParameters<double> parameters;

    HeatTransferCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG::Heat
