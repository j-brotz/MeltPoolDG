#pragma once
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  struct LevelSetCaseParameters : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      amr.add_parameters(prm);
      ls.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
      evapor.add_parameters(prm);
      material.add_parameters(prm);
      prm.enter_subsection("amr");
      {
        prm.add_parameter("strategy",
                          amr_strategy,
                          "Select the AMR strategy.",
                          dealii::Patterns::Selection("generic|refine_all_interface_cells"));
      }
      prm.leave_subsection();
    }

    void
    post(const std::string &parameter_filename) final
    {
      amr.post(base.global_refinements, false /*restart not supported*/);
      evapor.post(material, false /*does not matter*/);
      ls.post(base.fe, base.verbosity_level);
      output.post(time_stepping.time_step_size, parameter_filename);

      // check input parameters for validity
      ls.check_input_parameters(base.fe);
      profiling.check_input_parameters(time_stepping.time_step_size);
      evapor.check_input_parameters(material, ls.get_n_subdivisions());

      AssertThrow(evapor.evaporative_mass_flux_model ==
                    Evaporation::EvaporationModelType::analytical,
                  dealii::ExcMessage(
                    "For the level set case we only support to provide an analytical "
                    "function for the evaporation rate."));
    }

  public:
    BaseData                                  base;
    TimeIntegration::TimeSteppingData<number> time_stepping;
    AdaptiveMeshingData<number>               amr;
    LevelSetData<number>                      ls;
    OutputData<number>                        output;
    Profiling::ProfilingData<number>          profiling;
    std::string                               amr_strategy = "generic";
    Evaporation::EvaporationData<number>      evapor;
    MaterialData<number>                      material;
  };

  template <int dim, typename number>
  class LevelSetCase : public SimulationCaseBase<dim, number>
  {
  public:
    LevelSetCaseParameters<number> parameters;

    LevelSetCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG::LevelSet
