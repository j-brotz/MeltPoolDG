#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/evaporation/recoil_pressure_data.hpp>
#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/interface/base_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_parameters.hpp>
#include <meltpooldg/level_set/level_set_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/reinitialization/reinitialization_data.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

#include <fstream>
#include <iostream>
#include <string>

namespace MeltPoolDG
{
  using namespace dealii;

  BETTER_ENUM(DarcyDampingFormulation, char, implicit_formulation, explicit_formulation)

  BETTER_ENUM(TimeType, char, real, simulation)

  BETTER_ENUM(ConvectionStabilizationType,
              char,
              none,
              // streamline upwind Petrov-Galerkin stabilization
              SUPG)

  BETTER_ENUM(InterpolateMaterialParameterType,
              char,
              // no parameter interpolation typre specified; use the interpolation type specified
              // in MaterialData
              none,
              // sharp interpolation using a Heaviside function
              sharp,
              // interpolation using a smooth Heaviside function
              smooth,
              // reciprocal interpolation using a smooth Heaviside function
              reciprocal)

  struct AdaptiveMeshingData
  {
    bool         do_amr                       = false;
    bool         do_not_modify_boundary_cells = false;
    double       upper_perc_to_refine         = 0.0;
    double       lower_perc_to_coarsen        = 0.0;
    int          n_initial_refinement_cycles  = 0;
    int          every_n_step                 = 1;
    unsigned int max_grid_refinement_level    = 12;
    int          min_grid_refinement_level    = 1;
  };

  template <typename number = double>
  struct AdvectionDiffusionData
  {
    AdvectionDiffusionData()
    {
      linear_solver.solver_type         = LinearSolverType::GMRES;
      linear_solver.preconditioner_type = PreconditionerType::Diagonal;
    }

    number      diffusivity             = 0.0;
    std::string time_integration_scheme = "crank_nicolson";
    std::string implementation          = "meltpooldg";

    struct ConvectionStabilizationData
    {
      ConvectionStabilizationType type        = ConvectionStabilizationType::none;
      double                      coefficient = -1.0;
    } conv_stab;

    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;

    void
    post()
    {
      predictor.post();
    }
  };

  template <typename number = double>
  struct NormalVectorData
  {
    NormalVectorData()
    {
      linear_solver.solver_type         = LinearSolverType::CG;
      linear_solver.preconditioner_type = PreconditionerType::Diagonal;
    }

    number                   damping_scale_factor = 0.5;
    std::string              implementation       = "meltpooldg";
    unsigned int             verbosity_level      = 0;
    bool                     do_narrow_band       = false;
    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;
    number                   narrow_band_threshold = 0.9999999;

    void
    post()
    {
      predictor.post();
    }
  };

  template <typename number = double>
  struct CurvatureData
  {
    CurvatureData()
    {
      linear_solver.solver_type         = LinearSolverType::CG;
      linear_solver.preconditioner_type = PreconditionerType::Diagonal;
    }

    bool                     enable               = true;
    number                   damping_scale_factor = 0.0;
    std::string              implementation       = "meltpooldg";
    unsigned int             verbosity_level      = 0;
    bool                     do_narrow_band       = false;
    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;
    number                   narrow_band_threshold = 0.9999999;

    void
    post()
    {
      predictor.post();
    }
  };

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

  template <typename number = double>
  struct SurfaceTensionData
  {
    number surface_tension_coefficient                       = 0.0;
    number temperature_dependent_surface_tension_coefficient = 0.0;
    number reference_temperature                             = numbers::invalid_double;
    number coefficient_residual_fraction                     = 0.0;
    DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;
    bool                                        zero_surface_tension_in_solid = false;
    TimeStepLimitData<number>                   time_step_limit;
  };

  template <typename number = double>
  struct DarcyDampingData
  {
    number                  mushy_zone_morphology   = 0.0;
    number                  avoid_div_zero_constant = 1e-3;
    DarcyDampingFormulation formulation             = DarcyDampingFormulation::explicit_formulation;
  };

  template <typename number = double>
  struct ProfilingData
  {
    bool     enable               = false;
    double   write_time_step_size = 10.0;
    TimeType time_type            = TimeType::real;
  };

  template <typename number = double>
  struct RestartData
  {
    int         load                 = -1;
    int         save                 = -1;
    std::string directory            = "";
    std::string prefix               = "restart";
    double      write_time_step_size = 0.0;
    TimeType    time_type            = TimeType::real;
  };

  template <typename number = double>
  struct ParaviewData
  {
    bool                     do_output                      = false;
    bool                     do_user_defined_postprocessing = false;
    std::string              filename                       = "solution";
    std::string              directory                      = "./";
    int                      write_frequency                = 1;
    double                   write_time_step_size           = 0.0;
    bool                     print_boundary_id              = false;
    bool                     output_subdomains              = false;
    bool                     output_material_id             = false;
    int                      n_digits_timestep              = 4;
    int                      n_groups                       = 1;
    int                      n_patches                      = 0;
    bool                     write_higher_order_cells       = true;
    std::vector<std::string> output_variables               = {"all"};
  };

  /**
   * Utility function to print parameters from a given ParameterHandler object.
   */
  inline void
  print_parameters_external(const ParameterHandler &prm,
                            std::ostream           &pcout,
                            const bool              print_details)
  {
    prm.print_parameters(pcout,
                         print_details ? ParameterHandler::OutputStyle::JSON |
                                           ParameterHandler::OutputStyle::KeepDeclarationOrder :
                                         ParameterHandler::OutputStyle::ShortJSON);
  }

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
  struct Parameters
  {
    void
    process_parameters_file(ParameterHandler &prm, const std::string &parameter_filename);

    void
    print_parameters(ParameterHandler &prm, std::ostream &pcout, const bool print_details);

  private:
    void
    check_input_parameters() const;

    void
    check_for_file(const std::string &parameter_filename) const;

    void
         add_parameters(ParameterHandler &prm);
    bool parameters_read = false;

  public:
    BaseData<number>                                   base;
    TimeSteppingData<number>                           time_stepping;
    AdaptiveMeshingData                                amr;
    LevelSet::LevelSetData<number>                     ls;
    Reinitialization::ReinitializationData<number>     reinit;
    AdvectionDiffusionData<number>                     advec_diff;
    NormalVectorData<number>                           normal_vec;
    CurvatureData<number>                              curv;
    Heat::HeatData<number>                             heat;
    LaserData<number>                                  laser;
    RadiativeTransport::RadiativeTransportData<number> rte;
    MeltPoolData<number>                               mp;
    SurfaceTensionData<number>                         surface_tension;
    DarcyDampingData<number>                           darcy;
    Evaporation::RecoilPressureData<number>            recoil;
    Evaporation::EvaporationData<number>               evapor;
    MaterialData<number>                               material;
    ParaviewData<number>                               paraview;
    ProfilingData<number>                              profiling;
    RestartData<number>                                restart;
    Flow::AdafloWrapperParameters                      adaflo_params;
  };
} // namespace MeltPoolDG
