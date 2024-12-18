#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#include <meltpooldg/flow/flow_data.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/level_set/level_set_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/reinitialization/reinitialization_data.hpp>
#include <meltpooldg/utilities/amr_data.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/restart_data.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

#include <iostream>
#include <string>
#include <type_traits>

// TODO: move to case

namespace MeltPoolDG
{
  using namespace dealii;

  template <typename number = double>
  struct MeltPoolData
  {
    struct
    {
      bool   set_velocity_to_zero       = false;
      bool   do_not_reinitialize        = false;
      number solid_fraction_lower_limit = 1.0;
    } solid;
  };

  /**
   * Collection of all parameters of MeltPoolDG.
   *
   * @warning Parameters are read in order they are specified in the parameter
   * files. We exploit this behavior, e.g, in MaterialData, to allow to set
   * default parameters based on the user input and to override individual
   * entries subsequently. Please don't sort your input files and if you do be
   * aware that you might change the behavior!
   */
  template <typename number = double>
  struct Parameters : public ParametersBase
  {
  protected:
    void
    add_parameters(ParameterHandler &prm) final;

    void
    post(const std::string &parameter_filename) final;

  private:
    void
    check_input_parameters() const;


  public:
    BaseData                                           base;
    TimeSteppingData<number>                           time_stepping;
    AdaptiveMeshingData                                amr;
    LevelSet::LevelSetData<number>                     ls;
    Heat::HeatData<number>                             heat;
    Heat::LaserData<number>                            laser;
    RadiativeTransport::RadiativeTransportData<number> rte;
    MeltPoolData<number>                               mp;
    Flow::FlowData<number>                             flow;
    Evaporation::EvaporationData<number>               evapor;
    MaterialData<number>                               material;
    OutputData<number>                                 output;
    Profiling::ProfilingData<number>                   profiling;
    Restart::RestartData<number>                       restart;
    Flow::AdafloWrapperParameters                      adaflo_params;
  };
} // namespace MeltPoolDG
