#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_parameters.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/melt_pool/recoil_pressure_data.hpp>
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

  BETTER_ENUM(ProblemType,
              char,
              advection_diffusion,
              reinitialization,
              level_set,
              melt_pool,
              level_set_with_evaporation,
              heat_transfer,
              none)
  BETTER_ENUM(DarcyDampingFormulation, char, implicit_formulation, explicit_formulation)
  BETTER_ENUM(
    LaserHeatSourceModel,
    char,
    not_initialized, // must be specified by user
    Gauss,           // Gauss heat source distribution, see MeltPoolDG::Heat::LaserHeatSourceGauss
    Gusarov,         // Gusarov laser model, see MeltPoolDG::Heat::LaserHeatSourceGusarov
    Analytical, // analytical laser model, see MeltPoolDG::Heat::LaserAnalyticalTemperatureField
    uniform     // uniform laser model, see MeltPoolDG::Heat::LaserHeatSourceUniform
  )
  BETTER_ENUM(
    LaserImpactType,
    char,
    // volumetric heat source
    volumetric,
    // interfacial heat source; use continuum surface force modeling within the interface region
    interface,
    // interfacial heat source; evaluate integral as surface integral over the sharp interface
    interface_sharp)


  // evaporation specific @todo: move to own file evaporation_data.hpp
  BETTER_ENUM(
    EvaporationModelType,
    char,
    // prescribe a (time-dependent) function for an evaporative mass flux being constant
    // over the domain
    constant,
    // calculate the evaporative mass flux from the recoil pressure
    recoil_pressure,
    // calculate the evaporative mass flux according to the model proposed by Hardt & Wondra
    hardt_wondra)

  BETTER_ENUM(EvaporationLevelSetSourceTermType,
              char,
              // calculate the interface velocity by using the liquid velocity (H(phi)=1)
              interface_velocity_sharp_heavy,
              // calculate the interface velocity by using the value at the level set 0 iso contour
              interface_velocity_sharp,
              // calculate a divergence-free interface velocity and use it to advect the level set
              interface_velocity,
              // use the source term due to evaporation as right hand-side term
              rhs)

  BETTER_ENUM(InterfaceForceType, char, diffuse, sharp)

  // choose the particular predictor type for the nonlinear/linear solver
  BETTER_ENUM(PredictorType,
              char,
              // no predictor specified; use old value as initial guess
              none,
              // calculate the predictor by a linear combination from the two old solution vectors
              linear_extrapolation,
              // least squares projection (WIP)
              least_squares_projection)

  BETTER_ENUM(RestartTimeType, char, real, simulation)

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

  template <typename number = double>
  class PredictorData
  {
  public:
    PredictorData()
    {
      all.emplace_back(this);
    }

    PredictorType type                   = PredictorType::none;
    unsigned int  n_old_solution_vectors = 2;

    void
    add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("predictor");
      {
        prm.add_parameter("type", type, "Choose a predictor type.");
        prm.add_parameter("n old solutions",
                          n_old_solution_vectors,
                          "Choose the number of old solution vectors considered."
                          "This parameter is only relevant for least squares projection.");
      }
      prm.leave_subsection();
    }

    void
    set_default_values()
    {
      if (type == PredictorType::none)
        n_old_solution_vectors = 1;
      else if (type == PredictorType::linear_extrapolation)
        n_old_solution_vectors = 2;
    }

    static std::vector<PredictorData<number> *> all;
  };


  template <typename number>
  std::vector<PredictorData<number> *> PredictorData<number>::all;

  template <typename number = double>
  struct BaseData
  {
    std::string  application_name    = "none";
    ProblemType  problem_name        = ProblemType::advection_diffusion;
    unsigned int dimension           = 2;
    unsigned int global_refinements  = 1;
    unsigned int degree              = 1;
    int          n_q_points_1d       = -1;
    bool         do_print_parameters = true;
    bool         do_simplex          = false;
    number       gravity             = 0.0;
    unsigned int verbosity_level     = 0;
  };

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
  struct LevelSetData
  {
    bool        do_reinitialization     = true;
    int         n_initial_reinit_steps  = -1.0;
    std::string time_integration_scheme = "crank_nicolson";
    bool        do_curvature_correction = false;
    int         n_subdivisions          = 1;
    bool        do_localized_heaviside  = true;
    std::string implementation          = "meltpooldg";
    number      reinit_time_step_size   = -1.;
    number      tol_reinit              = std::numeric_limits<number>::min();
  };

  template <typename number = double>
  struct ReinitializationData
  {
    ReinitializationData()
    {
      linear_solver.solver_type         = LinearSolverType::CG;
      linear_solver.preconditioner_type = PreconditionerType::Diagonal;
    }

    unsigned int             max_n_steps          = 5; //@todo: move to LevelSetData
    number                   constant_epsilon     = -1.0;
    number                   scale_factor_epsilon = 0.5;
    std::string              modeltype            = "olsson2007";
    std::string              implementation       = "meltpooldg";
    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;
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
  };

  template <typename number = double>
  struct HeatData
  {
    HeatData()
    {
      linear_solver.solver_type         = LinearSolverType::GMRES;
      linear_solver.preconditioner_type = PreconditionerType::DiagonalReduced;
    }

    int                                         degree                   = -1;
    int                                         n_subdivisions           = 1;
    int                                         n_q_points_1d            = -1;
    number                                      emissivity               = 0.0;
    number                                      convection_coefficient   = 0.0;
    number                                      temperature_infinity     = 0.0;
    number                                      velocity                 = 0.0;
    bool                                        two_phase                = false;
    bool                                        solidification           = false;
    bool                                        enable_time_dependent_bc = false;
    NonlinearSolverData<number>                 nlsolve;
    LinearSolverData<number>                    linear_solver;
    DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;
    InterpolateMaterialParameterType            interpolate_rho_times_cp =
      InterpolateMaterialParameterType::none;
    InterpolateMaterialParameterType interpolate_k = InterpolateMaterialParameterType::none;
    PredictorData<number>            predictor;
  };



  template <typename number = double>
  struct LaserData
  {
    number              power            = 0.0;
    std::string         power_over_time  = "constant";
    number              power_start_time = 0.0;
    number              power_end_time   = 1.e12;
    std::vector<double> center; // default value will be set after parameters are read
    bool                do_move     = false;
    number              scan_speed  = 0.0;
    LaserImpactType     impact_type = LaserImpactType::volumetric;
    DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;
    LaserHeatSourceModel heat_source_model = LaserHeatSourceModel::not_initialized;
    struct GaussData
    {
      number laser_beam_radius   = 0.0;
      number absorptivity_liquid = 0.0;
      number absorptivity_gas    = 0.0;
    } gauss;
    struct GusarovData
    {
      number laser_beam_radius      = 0.0; // R
      number reflectivity           = 0.0; // rho
      number extinction_coefficient = 0.0; // beta
      number layer_thickness        = 0.0; // L
    } gusarov;
    struct AnalyticalData
    {
      number temperature_x_to_y_ratio = 1.0;
      number absorptivity_liquid      = 0.0;
      number absorptivity_gas         = 0.0;
      number max_temperature          = 0.0;
      number ambient_temperature      = 0.0;
    } analytical;
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

    struct Liquid
    {
      number melt_pool_radius = 0.0;
      number melt_pool_depth  = 0.0;
    } liquid;
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
  struct EvaporationData
  {
    number             evaporative_mass_flux_scale_factor = 1.0;
    std::string        evaporative_mass_flux              = "0.0";
    number             ls_value_liquid                    = 1.0;
    number             ls_value_gas                       = -1.0;
    InterfaceForceType formulation_source_term_continuity = InterfaceForceType::diffuse;
    InterfaceForceType formulation_source_term_heat       = InterfaceForceType::diffuse;
    std::string        formulation_evaporative_mass_flux_over_interface =
      "continuous"; // not needed if evaporation_model == "constant"
    EvaporationModelType              evaporation_model            = EvaporationModelType::constant;
    number                            coefficient                  = 0.0;
    unsigned int                      interface_value_n_iterations = 3;
    unsigned int                      line_integral_n_subdivisions_per_side = 10;
    unsigned int                      line_integral_n_subdivisions_MCA      = 1;
    EvaporationLevelSetSourceTermType level_set_source_term_type =
      EvaporationLevelSetSourceTermType::interface_velocity;
    bool do_level_set_pressure_gradient_interpolation = false;
  };

  template <typename number = double>
  struct ProfilingData
  {
    bool   enable               = false;
    int    write_frequency      = 1;
    double write_time_step_size = 0.0;
  };

  template <typename number = double>
  struct RestartData
  {
    int             load                 = -1;
    int             save                 = -1;
    std::string     directory            = "";
    std::string     prefix               = "restart";
    double          write_time_step_size = 0.0;
    RestartTimeType time_type            = RestartTimeType::real;
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
    bool                     do_initial_state               = true;
    bool                     print_boundary_id              = false;
    bool                     output_subdomains              = false;
    int                      n_digits_timestep              = 4;
    int                      n_groups                       = 1;
    int                      n_patches                      = 0;
    bool                     write_higher_order_cells       = true;
    std::vector<std::string> output_variables               = {"all"};
  };

  template <typename number = double>
  struct OutputData
  {
    bool        do_walltime              = 0;
    bool        do_compute_error         = 0;
    bool        do_compute_volume_output = 0;
    std::string filename_volume_output   = "my_volumes.txt";
  };

  /**
   * Utility function to print parameters from a given ParameterHandler object.
   */
  inline void
  print_parameters_external(const ParameterHandler &prm,
                            std::ostream &          pcout,
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
    BaseData<number>               base;
    TimeSteppingData<number>       time_stepping;
    AdaptiveMeshingData            amr;
    LevelSetData<number>           ls;
    ReinitializationData<number>   reinit;
    AdvectionDiffusionData<number> advec_diff;
    NormalVectorData<number>       normal_vec;
    CurvatureData<number>          curv;
    HeatData<number>               heat;
    LaserData<number>              laser;
    MeltPoolData<number>           mp;
    SurfaceTensionData<number>     surface_tension;
    DarcyDampingData<number>       darcy;
    RecoilPressureData<number>     recoil;
    EvaporationData<number>        evapor;
    MaterialData<number>           material;
    ParaviewData<number>           paraview;
    ProfilingData<number>          profiling;
    RestartData<number>            restart;
    OutputData<number>             output;
    Flow::AdafloWrapperParameters  adaflo_params;
  };
} // namespace MeltPoolDG
