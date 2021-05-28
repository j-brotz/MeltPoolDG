#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>

#include <fstream>
#include <iostream>
#include <string>

namespace MeltPoolDG
{
  using namespace dealii;

  template <typename number = double>
  struct SolverData
  {
    bool         do_matrix_free      = true;
    std::string  preconditioner_type = "Identity";
    std::string  solver_type         = "GMRES";
    unsigned int max_iterations      = 10000;
    number       rel_tolerance       = 1e-12;
  };

  template <typename number = double>
  struct NonlinearSolverData
  {
    int    max_nonlinear_iterations       = 10;
    number field_correction_tolerance     = 1e-10;
    number residual_tolerance             = 1e-9;
    int    max_nonlinear_iterations_alt   = 0;
    number field_correction_tolerance_alt = 1e-9;
    number residual_tolerance_alt         = 1e-8;
  };

  template <typename number = double>
  struct TimeSteppingData
  {
    number       start_time              = 0.0;
    number       end_time                = 1.0;
    number       time_step_size          = 0.01;
    unsigned int max_n_steps             = 1000000;
    std::string  time_integration_scheme = "none";
  };

  template <typename number = double>
  struct BaseData
  {
    std::string  application_name    = "none";
    std::string  problem_name        = "advection_diffusion";
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
    bool         do_not_modify_boundary_cells = true;
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
    bool        do_reinitialization     = false;
    int         n_initial_reinit_steps  = -1.0;
    number      artificial_diffusivity  = 0.0;
    std::string time_integration_scheme = "crank_nicolson";
    number      start_time              = 0.0;
    number      end_time                = 1.0;
    number      time_step_size          = 0.01;
    bool        enable_CFL_condition    = false;
    bool        do_curvature_correction = false;
    bool        do_matrix_free          = false;
    std::string implementation          = "meltpooldg";
  };

  template <typename number = double>
  struct ReinitializationData
  {
    unsigned int       max_n_steps          = 5;
    number             constant_epsilon     = -1.0;
    number             scale_factor_epsilon = 0.5;
    number             dtau                 = -1.0;
    std::string        modeltype            = "olsson2007";
    std::string        implementation       = "meltpooldg";
    SolverData<number> solver;
  };

  template <typename number = double>
  struct AdvectionDiffusionData
  {
    number       diffusivity             = 0.0;
    std::string  time_integration_scheme = "crank_nicolson";
    number       start_time              = 0.0;
    number       end_time                = 1.0;
    number       time_step_size          = 0.01;
    unsigned int max_n_steps             = 1000000;
    bool         do_matrix_free          = false;
    std::string  implementation          = "meltpooldg";
  };

  template <typename number = double>
  struct FlowData
  {
    int          velocity_degree                                   = -1;
    int          velocity_n_q_points_1d                            = -1;
    number       surface_tension_coefficient                       = 0.0;
    number       temperature_dependent_surface_tension_coefficient = 0.0;
    number       surface_tension_reference_temperature             = 0.0;
    number       surface_tension_coefficient_residual_fraction     = 0.0;
    std::string  solver_type                                       = "incompressible";
    number       start_time                                        = 0.0;
    number       end_time                                          = 1.0;
    number       time_step_size                                    = 0.05;
    unsigned int max_n_steps                                       = 1000000;
    std::string  variable_properties_over_interface                = "false";
  };

  template <typename number = double>
  struct NormalVectorData
  {
    number      damping_scale_factor = 0.5;
    bool        do_matrix_free       = false;
    std::string implementation       = "meltpooldg";
  };

  template <typename number = double>
  struct CurvatureData
  {
    number      damping_scale_factor = 0.0;
    bool        do_matrix_free       = false;
    std::string implementation       = "meltpooldg";
  };

  template <typename number = double>
  struct HeatData
  {
    number                      emissivity                         = 0.0;
    number                      convection_coefficient             = 0.0;
    number                      temperature_infinity               = 0.0;
    bool                        do_matrix_free                     = true;
    number                      velocity                           = 0.0;
    bool                        two_phase                          = false;
    bool                        variable_properties_over_interface = false;
    bool                        solidification                     = false;
    TimeSteppingData<number>    time_stepping;
    NonlinearSolverData<number> nlsolve;
    SolverData<number>          solver;
  };

  template <typename number = double>
  struct LaserData
  {
    number      power                              = 0.0;
    std::string power_over_time                    = "constant";
    number      power_start_time                   = 0.0;
    number      power_end_time                     = 1.e12;
    std::string center                             = "0,0,0";
    bool        do_move                            = false;
    number      scan_speed                         = 0.0;
    bool        variable_properties_over_interface = false;
    std::string heat_source_model                  = "Gusarov";
    struct GusarovData
    {
      number laser_beam_radius      = 0.0; // R
      number reflectivity           = 0.0; // rho
      number extinction_coefficient = 0.0; // beta
      number layer_thickness        = 0.0; // L
    } gusarov;
    struct GaussData
    {
      number laser_beam_radius = 0.0;
      number absorptivity      = 0.0;
    } gauss;
  };

  template <typename number = double>
  struct RecoilPressureData
  {
    number pressure_constant    = 0.0;
    number temperature_constant = 0.0;
  };

  template <typename number = double>
  struct MeltPoolData
  {
    std::string temperature_formulation        = "analytical";
    number      temperature_x_to_y_ratio       = 1.0;
    std::string melt_pool_center               = "not_initialized";
    bool        set_velocity_to_zero_in_solid  = false;
    bool        set_level_set_to_zero_in_solid = false;
    number      ambient_temperature            = 0.0;
    number      domain_x_min                   = 0.0;
    number      domain_y_min                   = 0.0;
    number      domain_x_max                   = 0.0;
    number      domain_y_max                   = 0.0;
    number      boiling_temperature            = 0.0;
    number      max_temperature                = 0.0;

    struct Liquid
    {
      number absorptivity     = 0.0;
      number melt_pool_radius = 0.0;
      number melt_pool_depth  = 0.0;
      number melting_point    = 0.0;
    } liquid;

    struct Gas
    {
      number absorptivity = 0.0;
    } gas;
  };

  template <typename number = double>
  struct EvaporationData
  {
    number      evaporative_mass_flux_scale_factor = 1.0;
    number      evaporative_mass_flux              = 0.0;
    number      ls_value_liquid                    = 1.0;
    number      ls_value_gas                       = -1.0;
    std::string formulation_source_term_continuity = "diffuse";
    std::string formulation_evaporative_mass_flux  = "constant"; // todo: recoil_pressure Schrage
    number      coefficient                        = 0.0;
    number      latent_heat_of_evaporation         = 0.0;
    number      molar_mass                         = 0.0;
    number      boiling_temperature                = 0.0;
  };

  template <typename number = double>
  struct MaterialData
  {
    /**
     * Default material. In case of two-phase flow; heaviside(level set) == 0
     */
    struct First
    {
      number capacity     = 0.0;
      number conductivity = 0.0;
      number density      = 0.0;
      number viscosity    = 0.0;
    } first;

    /**
     * Secondary material. In case of two-phase-flow; heaviside(level set) == 1
     */
    struct Second
    {
      number capacity     = 0.0;
      number conductivity = 0.0;
      number density      = 0.0;
      number viscosity    = 0.0;
    } second;

    /**
     * Solid material.
     */
    struct Solid
    {
      number capacity     = 0.0;
      number conductivity = 0.0;
      number density      = 0.0;
    } solid;

    number solidus_temperature  = 0.0;
    number liquidus_temperature = 0.0;
  };

  template <typename number = double>
  struct ParaviewData
  {
    bool        do_output           = false;
    std::string filename            = "solution";
    std::string directory           = "./";
    int         write_frequency     = 1;
    bool        do_initial_state    = true;
    bool        print_levelset      = true;
    bool        print_normal_vector = false;
    bool        print_curvature     = false;
    bool        print_advection     = false;
    bool        print_exactsolution = false;
    bool        print_boundary_id   = false;
    int         n_digits_timestep   = 4;
    int         n_groups            = 1;
  };

  template <typename number = double>
  struct OutputData
  {
    bool        do_walltime              = 0;
    bool        do_compute_error         = 0;
    bool        do_compute_volume_output = 0;
    std::string filename_volume_output   = "my_volumes.txt";
  };

  template <typename number = double>
  struct Parameters
  {
    void
    process_parameters_file(const std::string &parameter_filename);

    void
    print_parameters(std::ostream &pcout);

    void
    check_for_file(const std::string &parameter_filename) const;

    void
    add_parameters();

    ParameterHandler prm;

    BaseData<number>               base;
    AdaptiveMeshingData            amr;
    LevelSetData<number>           ls;
    ReinitializationData<number>   reinit;
    AdvectionDiffusionData<number> advec_diff;
    FlowData<number>               flow;
    NormalVectorData<number>       normal_vec;
    CurvatureData<number>          curv;
    HeatData<number>               heat;
    LaserData<number>              laser;
    MeltPoolData<number>           mp;
    RecoilPressureData<number>     recoil;
    EvaporationData<number>        evapor;
    MaterialData<number>           material;
    ParaviewData<number>           paraview;
    OutputData<number>             output;
    Flow::AdafloWrapperParameters  adaflo_params;
  };
} // namespace MeltPoolDG
