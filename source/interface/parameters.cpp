#include <meltpooldg/interface/parameters.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  Parameters<number>::process_parameters_file(const std::string &parameter_filename)
  {
    add_parameters();

    check_for_file(parameter_filename);

    std::ifstream file;
    file.open(parameter_filename);

    if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "json")
      prm.parse_input_from_json(file, true);
    else if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "prm")
      prm.parse_input(parameter_filename);
    else
      AssertThrow(false, ExcMessage("Parameterhandler cannot handle current file ending"));
    /*
     *  set the number of quadrature points in 1d
     */
    base.n_q_points_1d = (base.n_q_points_1d < 1) ? base.degree + 1 : base.n_q_points_1d;
    /*
     *  set the min grid refinement level if not user-specified
     */
    if (amr.min_grid_refinement_level == 1)
      amr.min_grid_refinement_level = base.global_refinements;

    /*
     *  set the number of initial reinitialization steps equal to the number of reinit steps
     *  if no value is provided
     */
    if (ls.do_reinitialization && ls.n_initial_reinit_steps < 0.0)
      ls.n_initial_reinit_steps = reinit.max_n_steps;
    /*
     *  set the melt pool center if not specified
     */
    if (mp.melt_pool_center == "not_initialized")
      mp.melt_pool_center = laser.center;
    /*
     *  set the maximum temperature of the melt pool if not specified
     */
    if (mp.max_temperature < mp.boiling_temperature)
      mp.max_temperature = mp.boiling_temperature + 500;
    /*
     *  check if level set assignment of gaseous/liquid phase is done correctly
     */
    if (evapor.ls_value_liquid == evapor.ls_value_gas)
      AssertThrow(
        false, ExcMessage("Parameterhandler: ls value liquid must not be equal to ls value gas."));
      /*
       *  parameters for adaflo
       */
#ifdef MELT_POOL_DG_WITH_ADAFLO

    if (base.problem_name == "two_phase_flow" || base.problem_name == "melt_pool" ||
        base.problem_name == "two_phase_flow_with_evaporation" ||
        base.problem_name == "melt_pool_with_evaporation" ||
        base.problem_name == "two_phase_flow_with_heat_transfer")
      {
        adaflo_params.parse_parameters(parameter_filename);

        AssertThrow(adaflo_params.params.density == 1.0, // 1.0 is the default value from adaflo
                    ExcMessage(
                      "It seems that you specified the density parameter "
                      "within the adaflo section, which is ignored by MeltPoolDG. "
                      "Please use the >material: material first density:< section instead. "));

        AssertThrow(adaflo_params.params.viscosity == 1.0, // 1.0 is the default value from adaflo
                    ExcMessage(
                      "It seems that you specified the viscosity parameter "
                      "within the adaflo section, which is ignored by MeltPoolDG. "
                      "Please use the >material: material first density:< section instead. "));

        if (base.problem_name == "melt_pool_with_evaporation" ||
            base.problem_name == "two_phase_flow_with_evaporation")
          {
            if (evapor.formulation_source_term_continuity != "sharp")
              {
                AssertThrow(
                  adaflo_params.params.beta_convective_term_momentum_balance == 0,
                  ExcMessage(
                    "For the consideration of phase change, the convective "
                    "formulation of the momentum balance in the Navier-Stokes equations "
                    "must be chosen: Navier-Stokes: adaflo: Navier-Stokes: {formulation convective "
                    "term momentum balance: convective }"));

                // AssertThrow(flow.variable_properties_over_interface ==
                //"consistent_with_evaporation",
                // ExcMessage(
                //"For the consideration of phase change, the density "
                //"has to be interpolated consistently with the continuity equation "
                //"including phase change."));
              }
          }
        // WARNING: by setting the differences to a non-zero value we force
        //   adaflo to assume that we are running a simulation with variable
        //   coefficients, i.e., it allocates memory for the data structures
        //   variable_densities and variable_viscosities, which are accessed
        //   during NavierStokesMatrix::begin_densities() and
        //   NavierStokesMatrix::begin_viscosity(). However, we do not actually
        //   use these values, since we fill the density and viscosity
        //   differently.
        adaflo_params.params.density_diff   = 1.0;
        adaflo_params.params.viscosity_diff = 1.0;

        if (material.first.density > 0.0)
          {
            // adaflo assumes the parameter density to be the one of heaviside == 0
            adaflo_params.params.density      = material.first.density;
            adaflo_params.params.density_diff = (material.second.density > 0.0) ?
                                                  material.first.density - material.second.density :
                                                  0.0;
          }
        if (material.first.viscosity > 0.0)
          {
            // adaflo assumes the parameter viscosity to be the one of heaviside == 0
            adaflo_params.params.viscosity = material.first.viscosity;
            adaflo_params.params.viscosity_diff =
              (material.second.viscosity > 0.0) ?
                material.first.viscosity - material.second.viscosity :
                0.0;
          }

        flow.velocity_degree        = (flow.velocity_degree > 0.0) ? flow.velocity_degree :
                                                                     adaflo_params.params.velocity_degree;
        flow.velocity_n_q_points_1d = (flow.velocity_n_q_points_1d < 1) ?
                                        flow.velocity_degree + 1 :
                                        flow.velocity_n_q_points_1d;

        /// synchronize time stepping schemes
        adaflo_params.params.start_time           = flow.start_time;
        adaflo_params.params.end_time             = flow.end_time;
        adaflo_params.params.time_step_size_start = flow.time_step_size;
        adaflo_params.params.time_step_size_min   = flow.time_step_size;
        adaflo_params.params.time_step_size_max   = flow.time_step_size;
        adaflo_params.params.use_simplex_mesh     = base.do_simplex;
      }
#endif
  }

  template <typename number>
  void
  Parameters<number>::print_parameters(std::ostream &pcout)
  {
    if (base.do_print_parameters)
      prm.print_parameters(pcout, ParameterHandler::OutputStyle::Text);
  }

  template <typename number>
  void
  Parameters<number>::check_for_file(const std::string &parameter_filename) const
  {
    std::ifstream parameter_file(parameter_filename.c_str());

    if (!parameter_file)
      {
        parameter_file.close();

        std::ostringstream message;
        message << "Input parameter file <" << parameter_filename
                << "> not found. Please make sure the file exists!" << std::endl;

        AssertThrow(false, ExcMessage(message.str().c_str()));
      }
  }

  template <typename number>
  void
  Parameters<number>::add_parameters()
  {
    /*
     *    base
     */
    prm.enter_subsection("base");
    {
      prm.add_parameter(
        "application name",
        base.application_name,
        "Sets the base name for the application that will be fed to the problem type.");
      prm.add_parameter(
        "problem name",
        base.problem_name,
        "Sets the base name for the problem that should be solved.",
        Patterns::Selection(
          "advection_diffusion|reinitialization|level_set|two_phase_flow|melt_pool"
          "|level_set_with_evaporation|two_phase_flow_with_evaporation|"
          "melt_pool_with_evaporation|heat_transfer|two_phase_flow_with_heat_transfer"));
      prm.add_parameter("dimension", base.dimension, "Defines the dimension of the problem");
      prm.add_parameter("global refinements",
                        base.global_refinements,
                        "Defines the number of initial global refinements");
      prm.add_parameter("degree", base.degree, "Defines the interpolation degree");
      prm.add_parameter("n q points 1d",
                        base.n_q_points_1d,
                        "Defines the number of quadrature points");
      prm.add_parameter("do print parameters",
                        base.do_print_parameters,
                        "Sets this parameter to true to list parameters in output");
      prm.add_parameter("do simplex", base.do_simplex, "Use simplices");
      prm.add_parameter("gravity", base.gravity, "Set the value for the gravity");
      prm.add_parameter(
        "verbosity level",
        base.verbosity_level,
        "Sets the maximum verbosity level of the console output. Set this parameter to 0 in case of test files.");
    }
    prm.leave_subsection();
    /*
     *    adaptive meshing
     */
    prm.enter_subsection("adaptive meshing");
    {
      prm.add_parameter("do amr",
                        amr.do_amr,
                        "Sets this parameter to true to activate adaptive meshing");
      prm.add_parameter("do not modify boundary cells",
                        amr.do_not_modify_boundary_cells,
                        "Sets this parameter to true to not refine/coarsen along boundaries.");
      prm.add_parameter("upper perc to refine",
                        amr.upper_perc_to_refine,
                        "Defines the (upper) percentage of elements that should be refined");
      prm.add_parameter("lower perc to coarsen",
                        amr.lower_perc_to_coarsen,
                        "Defines the (lower) percentage of elements that should be coarsened");
      prm.add_parameter(
        "max grid refinement level",
        amr.max_grid_refinement_level,
        "Defines the number of maximum refinement steps one grid cell will be undergone.");
      prm.add_parameter(
        "min grid refinement level",
        amr.min_grid_refinement_level,
        "Defines the number of minimum refinement steps one grid cell will be undergone.");
      prm.add_parameter("n initial refinement cycles",
                        amr.n_initial_refinement_cycles,
                        "Defines the number of initial refinements.");
      prm.add_parameter("every n step",
                        amr.every_n_step,
                        "Defines at every nth step the amr should be performed.");
    }
    prm.leave_subsection();
    /*
     *   advection diffusion
     */
    prm.enter_subsection("advection diffusion");
    {
      prm.add_parameter("advec diff diffusivity",
                        advec_diff.diffusivity,
                        "Defines the diffusivity for the advection diffusion equation ");
      prm.add_parameter("advec diff time integration scheme",
                        advec_diff.time_integration_scheme,
                        "Determines the time integration scheme.",
                        Patterns::Selection("explicit_euler|implicit_euler|crank_nicolson|bdf_2"));
      prm.add_parameter("advec diff start time",
                        advec_diff.start_time,
                        "Defines the start time for the solution of the levelset problem");
      prm.add_parameter("advec diff end time",
                        advec_diff.end_time,
                        "Sets the end time for the solution of the advection diffusion problem");
      prm.add_parameter(
        "advec diff time step size",
        advec_diff.time_step_size,
        "Sets the step size for time stepping. For non-uniform "
        "time stepping, this parameter determines the size of the first time step.");
      prm.add_parameter("advec diff max n steps",
                        advec_diff.max_n_steps,
                        "Sets the maximum number of advection diffusion steps");
      prm.add_parameter(
        "advec diff do matrix free",
        advec_diff.do_matrix_free,
        "Set this parameter if a matrix free solution procedure should be performed.");
      prm.add_parameter(
        "advec diff implementation",
        advec_diff.implementation,
        "Choose the corresponding implementation of the advection diffusion operation.",
        Patterns::Selection("meltpooldg|adaflo"));
    }
    prm.leave_subsection();

    /*
     *   levelset
     */
    prm.enter_subsection("levelset");
    {
      prm.add_parameter("ls artificial diffusivity",
                        ls.artificial_diffusivity,
                        "Defines the artificial diffusivity for the level set transport equation");

      prm.add_parameter("ls do reinitialization",
                        ls.do_reinitialization,
                        "Defines if reinitialization of level set function is activated");
      prm.add_parameter(
        "ls n initial reinit steps",
        ls.n_initial_reinit_steps,
        "Defines the number of initial reinitialization steps of the level set function.");
      prm.add_parameter("ls time integration scheme",
                        ls.time_integration_scheme,
                        "Determines the time integration scheme.",
                        Patterns::Selection("explicit_euler|implicit_euler|crank_nicolson"));
      prm.add_parameter("ls start time",
                        ls.start_time,
                        "Defines the start time for the solution of the levelset problem");
      prm.add_parameter("ls end time",
                        ls.end_time,
                        "Sets the end time for the solution of the levelset problem");
      prm.add_parameter(
        "ls time step size",
        ls.time_step_size,
        "Sets the step size for time stepping. For non-uniform "
        "time stepping, this parameter determines the size of the first time step.");
      prm.add_parameter("ls enable CFL condition",
                        ls.enable_CFL_condition,
                        "Enables to dynamically adapt the time step to meet the CFL condition"
                        " in case of explicit time integration.");
      prm.add_parameter(
        "ls do matrix free",
        ls.do_matrix_free,
        "Set this parameter if a matrix free solution procedure should be performed.");
      prm.add_parameter(
        "ls do curvature correction",
        ls.do_curvature_correction,
        "Set this parameter to true if in areas outside the interface region a correction "
        "of the curvature values should be applied. This parameter can be helpful to avoid "
        "numerical instabilities.");
      prm.add_parameter("ls implementation",
                        ls.implementation,
                        "Choose the corresponding implementation of the ls operation.",
                        Patterns::Selection("meltpooldg|adaflo"));
    }
    prm.leave_subsection();

    /*
     *   reinitialization
     */
    prm.enter_subsection("reinitialization");
    {
      prm.add_parameter("reinit max n steps",
                        reinit.max_n_steps,
                        "Sets the maximum number of reinitialization steps");
      prm.add_parameter("reinit constant epsilon",
                        reinit.constant_epsilon,
                        "Defines the length parameter of the level set function to be constant and"
                        "not to dependent on the mesh size (default: -1.0 i.e. grid size dependent"
                        "which can be controlled by reinit_epsilon_scale_factor");
      prm.add_parameter(
        "reinit scale factor epsilon",
        reinit.scale_factor_epsilon,
        "Defines the scaling factor of the diffusion parameter in the reinitialization "
        "equation; the scaling factor is multipled by the mesh size (default: 0.5 i.e. eps=0.5*h_min");
      prm.add_parameter(
        "reinit dtau",
        reinit.dtau,
        "Defines the time step size of the reinitialization to be constant and"
        "not to dependent on the mesh size (default: -1.0 i.e. grid size dependent");
      prm.add_parameter("reinit modeltype",
                        reinit.modeltype,
                        "Sets the type of reinitialization model that should be used.");
      prm.add_parameter(
        "reinit do matrix free",
        reinit.solver.do_matrix_free,
        "Set this parameter if a matrix free solution procedure should be performed.");
      prm.add_parameter(
        "reinit solver type",
        reinit.solver.solver_type,
        "Set this parameter for choosing a solver type. At the moment GMRES or CG solvers "
        " are supported");
      prm.add_parameter("reinit preconditioner type",
                        reinit.solver.preconditioner_type,
                        "Set this parameter for choosing a preconditioner type",
                        Patterns::Selection("AMG|Identity|ILU"));
      prm.add_parameter(
        "reinit max iterations",
        reinit.solver.max_iterations,
        "Set the maximum number of iterations for solving the linear system of equations.");
      prm.add_parameter(
        "reinit rel tolerance",
        reinit.solver.rel_tolerance,
        "Set the relative tolerance for a successful solution of the linear system of equations.");
      prm.add_parameter(
        "reinit implementation",
        reinit.implementation,
        "Choose the corresponding implementation of the reinitialization operation.",
        Patterns::Selection("meltpooldg|adaflo"));
    }
    prm.leave_subsection();
    /*
     *   normal vector
     */
    prm.enter_subsection("normal vector");
    {
      prm.add_parameter(
        "normal vec damping scale factor",
        normal_vec.damping_scale_factor,
        "normal vector computation: damping = cell_size * normal_vec_damping_scale_factor");
      prm.add_parameter(
        "normal vec do matrix free",
        normal_vec.do_matrix_free,
        "Set this parameter if a matrix free solution procedure should be performed.");
      prm.add_parameter("normal vec implementation",
                        normal_vec.implementation,
                        "Choose the corresponding implementation of the normal vector operation.",
                        Patterns::Selection("meltpooldg|adaflo"));
    }
    prm.leave_subsection();
    /*
     *   curvature
     */
    prm.enter_subsection("curvature");
    {
      prm.add_parameter("curv damping scale factor",
                        curv.damping_scale_factor,
                        "curvature computation: damping = cell_size * curv_damping_scale_factor");
      prm.add_parameter(
        "curv do matrix free",
        curv.do_matrix_free,
        "Set this parameter if a matrix free solution procedure should be performed.");
      prm.add_parameter("curv implementation",
                        curv.implementation,
                        "Choose the corresponding implementation of the curvature operation.",
                        Patterns::Selection("meltpooldg|adaflo"));
    }
    prm.leave_subsection();
    /*
     *   flow
     */
    prm.enter_subsection("flow");
    {
      prm.add_parameter("flow velocity degree",
                        flow.velocity_degree,
                        "velocity degree of the flow field");
      prm.add_parameter("flow n q points 1d",
                        flow.velocity_n_q_points_1d,
                        "number of 1d quadrature points for the velocity field of the flow");
      prm.add_parameter("flow surface tension coefficient",
                        flow.surface_tension_coefficient,
                        "constant coefficient for calculating surface tension");
      prm.add_parameter("flow temperature dependent surface tension coefficient",
                        flow.temperature_dependent_surface_tension_coefficient,
                        "temperature dependent coefficient for calculating temperetaure-dependent "
                        "surface tension (Marangoni convection)");
      prm.add_parameter("flow surface tension reference temperature",
                        flow.surface_tension_reference_temperature,
                        "Reference temperature for calculating surface tension");
      prm.add_parameter(
        "flow surface tension coefficient residual fraction",
        flow.surface_tension_coefficient_residual_fraction,
        "Define the minimum fraction of the constant surface tension reference value "
        "that can be reached.");
      prm.add_parameter("flow solver type", flow.solver_type, "solver type of the flow problem");
      prm.add_parameter("flow start time",
                        flow.start_time,
                        "Defines the start time for the solution of the levelset problem");
      prm.add_parameter("flow end time",
                        flow.end_time,
                        "Sets the end time for the solution of the levelset problem");
      prm.add_parameter("flow time step size",
                        flow.time_step_size,
                        "Sets the step size for time stepping. For non-uniform "
                        "time stepping, this parameter determines the size of the first "
                        "time step.");
      prm.add_parameter("flow max n steps",
                        flow.max_n_steps,
                        "Sets the maximum number of flow steps");
      prm.add_parameter(
        "flow variable properties over interface",
        flow.variable_properties_over_interface,
        "Set this parameter to interpolate the flow properties over the interface smoothly.",
        Patterns::Selection("false|true|consistent_with_evaporation"));
    }
    prm.leave_subsection();
    /*
     *   heat
     */
    prm.enter_subsection("heat");
    {
      prm.add_parameter("heat convection coefficient",
                        heat.convection_coefficient,
                        "Convection coefficient for the radiative boundary condition");
      prm.add_parameter("heat emissivity",
                        heat.emissivity,
                        "Emissivity for the radiative boundary condition");
      prm.add_parameter("heat temperature infinity",
                        heat.temperature_infinity,
                        "Infinity temperature for the conductive and radiative boundary condition");
      prm.add_parameter(
        "heat do matrix free",
        heat.do_matrix_free,
        "Set this parameter if a matrix free solution procedure should be performed.");
      prm.add_parameter(
        "heat solver type",
        heat.solver.solver_type,
        "Set this parameter for choosing a solver type. At the moment GMRES or CG solvers "
        " are supported");
      prm.add_parameter("heat solver preconditioner type",
                        heat.solver.preconditioner_type,
                        "Set this parameter for choosing a preconditioner type",
                        Patterns::Selection(
                          "Identity|AMG|AMGReduced|ILU|ILUReduced|Diagonal|DiagonalReduced"));
      prm.add_parameter(
        "heat solver max iterations",
        heat.solver.max_iterations,
        "Set the maximum number of iterations for solving the linear system of equations.");
      prm.add_parameter(
        "heat solver rel tolerance",
        heat.solver.rel_tolerance,
        "Set the relative tolerance for a successful solution of the linear system of equations.");
      prm.add_parameter("heat nlsolve max nonlinear iterations",
                        heat.nlsolve.max_nonlinear_iterations,
                        "Set the number of maximum nonlinear iterations with standard tolerances.");
      prm.add_parameter(
        "heat nlsolve field correction tolerance",
        heat.nlsolve.field_correction_tolerance,
        "Set the tolerance for the maximum allowed correction of the unknown field.");
      prm.add_parameter(
        "heat nlsolve residual tolerance",
        heat.nlsolve.residual_tolerance,
        "Set the tolerance for the maximum allowed residual of the nonlinear system.");
      prm.add_parameter(
        "heat nlsolve max nonlinear iterations alt",
        heat.nlsolve.max_nonlinear_iterations_alt,
        "Set the number of maximum nonlinear iterations with alternative tolerances.");
      prm.add_parameter(
        "heat nlsolve field correction tolerance alt",
        heat.nlsolve.field_correction_tolerance_alt,
        "Set the alternative tolerance for the maximum allowed correction of the unknown field.");
      prm.add_parameter(
        "heat nlsolve residual tolerance alt",
        heat.nlsolve.residual_tolerance_alt,
        "Set the alternative tolerance for the maximum allowed residual of the nonlinear system.");
      prm.add_parameter("heat start time",
                        heat.time_stepping.start_time,
                        "Defines the start time for the solution of the heat problem");
      prm.add_parameter("heat end time",
                        heat.time_stepping.end_time,
                        "Sets the end time for the solution of the heat problem");
      prm.add_parameter("heat time step size",
                        heat.time_stepping.time_step_size,
                        "Sets the step size for time stepping. For non-uniform "
                        "time stepping, this parameter determines the size of the first "
                        "time step.");
      prm.add_parameter("heat max n steps",
                        heat.time_stepping.max_n_steps,
                        "Sets the maximum number of time steps");
      prm.add_parameter("heat velocity", heat.velocity, "Velocity.");
      prm.add_parameter("heat two phase", heat.two_phase, "Set this parameter for two phase flow.");
      prm.add_parameter(
        "heat variable properties over interface",
        heat.variable_properties_over_interface,
        "Set this parameter to true to smear the material properites over the two-phase interface.");
      prm.add_parameter("heat solidification",
                        heat.solidification,
                        "Set this parameter to true to consider solidification.");
    }
    prm.leave_subsection();
    /*
     *   laser
     */
    prm.enter_subsection("laser");
    {
      prm.add_parameter("laser power", laser.power, "Intensity of the laser");
      prm.add_parameter("laser power over time",
                        laser.power_over_time,
                        "Temporal distribution of the laser power",
                        Patterns::Selection("constant|ramp"));
      prm.add_parameter("laser power start time",
                        laser.power_start_time,
                        "In case of time-dependent laser power: activation time of laser.");
      prm.add_parameter("laser power end time",
                        laser.power_end_time,
                        "In case of time-dependent laser power: end time of laser.");
      prm.add_parameter("laser center",
                        laser.center,
                        "Center coordinates of the laser beam on the interface melt/gas.");
      prm.add_parameter("laser scan speed",
                        laser.scan_speed,
                        "Scan speed of the laser (in case of an analytical temperature field).");
      prm.add_parameter(
        "laser variable properties over interface",
        laser.variable_properties_over_interface,
        "Set this parameter to true to interpolate the thermal properties over the interface smoothly.");
      prm.add_parameter(
        "laser do move",
        laser.do_move,
        "Set this parameter to true to move the laser in x-direction with the given parameter scan speed "
        "(in case of an analytical temperature field).");
      prm.add_parameter("laser heat source model",
                        laser.heat_source_model,
                        "Laser heat source model.",
                        Patterns::Selection("Gusarov|Gauss"));
      prm.add_parameter("laser gusarov laser beam radius",
                        laser.gusarov.laser_beam_radius,
                        "Laser beam radius.");
      prm.add_parameter("laser gusarov reflectivity",
                        laser.gusarov.reflectivity,
                        "Reflectivity of the material.");
      prm.add_parameter("laser gusarov extinction coefficient",
                        laser.gusarov.extinction_coefficient,
                        "Extinction coefficient in [1/m].");
      prm.add_parameter("laser gusarov layer thickness",
                        laser.gusarov.layer_thickness,
                        "Layer thickness");
      prm.add_parameter(
        "laser impact",
        laser.impact,
        "Laser impact model. volumetric: volumetric heat source | surface: surface heat source at two-phase interface.",
        Patterns::Selection("volumetric|interface"));
      prm.add_parameter("laser gauss laser beam radius",
                        laser.gauss.laser_beam_radius,
                        "Laser beam radius.");
      prm.add_parameter("laser gauss absorptivity",
                        laser.gauss.absorptivity,
                        "Laser energy absorptivity.");
    }
    prm.leave_subsection();
    /*
     *   recoil pressure
     */
    prm.enter_subsection("recoil pressure");
    {
      prm.add_parameter("recoil pressure constant",
                        recoil.pressure_constant,
                        "Pressure constant for the recoil pressure model.");
      prm.add_parameter("recoil temperature constant",
                        recoil.temperature_constant,
                        "Temperature constant for the recoil pressure model.");
    }
    prm.leave_subsection();
    /*
     *   melt pool
     */
    prm.enter_subsection("melt pool");
    {
      prm.add_parameter("mp temperature formulation",
                        mp.temperature_formulation,
                        "Definition type of the temperature field: "
                        "(1) analytical expression (2) solve heat equation (not implemented yet)",
                        Patterns::Selection("analytical"));
      prm.add_parameter("mp temperature x to y ratio",
                        mp.temperature_x_to_y_ratio,
                        "This factor scales the analytical temperature field to be anisotropic.");
      prm.add_parameter("mp melt pool center",
                        mp.melt_pool_center,
                        "Center coordinates of the melt pool ellipse/parabola. If no value is "
                        "provided it will be set equally to the laser center");
      prm.add_parameter("mp domain x min",
                        mp.domain_x_min,
                        "minimum x coordinate of simulation domain");
      prm.add_parameter("mp domain y min",
                        mp.domain_y_min,
                        "minimum y coordinate of simulation domain");
      prm.add_parameter("mp domain x max",
                        mp.domain_x_max,
                        "maximum x coordinate of simulation domain");
      prm.add_parameter("mp domain y max",
                        mp.domain_y_max,
                        "maximum y coordinate of simulation domain");
      prm.add_parameter(
        "mp set velocity to zero in solid",
        mp.set_velocity_to_zero_in_solid,
        "Set this parameter to true to constrain the flow velocity in the solid domain.");
      prm.add_parameter(
        "mp set level set to zero in solid",
        mp.set_level_set_to_zero_in_solid,
        "Set this parameter to true to constrain the level set in the solid domain.");
      prm.add_parameter("mp ambient temperature",
                        mp.ambient_temperature,
                        "Ambient temperature in the inert gas.");
      prm.add_parameter("mp boiling temperature",
                        mp.boiling_temperature,
                        "Boiling temperature of the melt.");
      prm.add_parameter(
        "mp max temperature",
        mp.max_temperature,
        "Maximum temperature arising in the melt pool. If this temperature is lower than the boiling"
        " temperature, this value is corrected to correspond to the boiling temperature + 500 K.");
      prm.add_parameter("mp liquid absorptivity",
                        mp.liquid.absorptivity,
                        "Absorptivity of the liquid part of domain");
      prm.add_parameter("mp liquid melt pool radius",
                        mp.liquid.melt_pool_radius,
                        "Set the radius of the liquid parts of the melt pool ellipse "
                        " or the width of the parabola");
      prm.add_parameter("mp liquid melt pool depth",
                        mp.liquid.melt_pool_depth,
                        "Set the depth of the liquid parts of the melt pool ellipse");
      prm.add_parameter("mp liquid melting point",
                        mp.liquid.melting_point,
                        "Melting point of the liquid part of domain");
      prm.add_parameter("mp gas absorptivity",
                        mp.gas.absorptivity,
                        "Absorptivity of the gaseous part of domain");
    }

    prm.leave_subsection();
    /*
     *  evaporation
     */
    prm.enter_subsection("evaporation");
    {
      prm.add_parameter("evapor evaporative mass flux scale factor",
                        evapor.evaporative_mass_flux_scale_factor,
                        "Scale factor for the evaporative flux");
      prm.add_parameter("evapor evaporative mass flux",
                        evapor.evaporative_mass_flux,
                        "Mass flux due to evaporation (SI unit in kg/m²s).");
      prm.add_parameter("evapor ls value liquid",
                        evapor.ls_value_liquid,
                        "Set the level set value corresponding to the liquid domain.",
                        Patterns::Selection("1|-1|1.|-1.|1.0|-1.0"));
      prm.add_parameter("evapor ls value gas",
                        evapor.ls_value_gas,
                        "Set the level set value corresponding to the gaseous domain.",
                        Patterns::Selection("1|-1|1.|-1.|1.0|-1.0"));
      prm.add_parameter("evapor formulation source term continuity",
                        evapor.formulation_source_term_continuity,
                        "Select how the additional source term due to evaporation in the"
                        " continuity equation is computed.",
                        Patterns::Selection("diffuse|sharp"));
      // @todo must be modified
      prm.add_parameter(
        "evapor formulation evaporative mass flux over interface",
        evapor.formulation_evaporative_mass_flux_over_interface,
        "Choose the formulation how the (local) evaporative mass flux will be converted to a DoF vector."
        "will be calculated.",
        Patterns::Selection("continuous|interface value|line integral"));
      prm.add_parameter("evapor evaporation model",
                        evapor.evaporation_model,
                        "Choose the formulation how the evaporative mass flux mDot (kg/(m2s)) "
                        "will be calculated.",
                        Patterns::Selection("constant|recoil pressure|Hardt Wondra"));
      prm.add_parameter("evapor coefficient", evapor.coefficient, "Evaporation coefficient.");
      prm.add_parameter("evapor latent heat of evaporation",
                        evapor.latent_heat_of_evaporation,
                        "Latent heat of evaporation (J/kg).");
      prm.add_parameter("evapor molar mass", evapor.molar_mass, "Molar mass (mol/kg).");
      prm.add_parameter("evapor boiling temperature",
                        evapor.boiling_temperature,
                        "Boiling temperature (K).");
    }
    prm.leave_subsection();
    /*
     *  material
     */
    prm.enter_subsection("material");
    {
      prm.add_parameter(
        "material first capacity",
        material.first.capacity,
        "capacity of the primary material (in case of two-phases corresponding to level-set = -1)");
      prm.add_parameter(
        "material first conductivity",
        material.first.conductivity,
        "conductivity of the primary material (in case of two-phases corresponding to level-set = -1)");
      prm.add_parameter(
        "material first density",
        material.first.density,
        "density of the primary material (in case of two-phases corresponding to level-set = -1)");
      prm.add_parameter(
        "material first viscosity",
        material.first.viscosity,
        "viscosity of the primary material (in case of two-phases corresponding to level-set = -1)");
      prm.add_parameter("material second capacity",
                        material.second.capacity,
                        "capacity of the secondary material (level-set = 1)");
      prm.add_parameter("material second conductivity",
                        material.second.conductivity,
                        "conductivity of the secondary material (level-set = 1)");
      prm.add_parameter("material second density",
                        material.second.density,
                        "density of the secondary material (level-set = 1)");
      prm.add_parameter("material second viscosity",
                        material.second.viscosity,
                        "viscosity of the secondary material (level-set = 1)");
      prm.add_parameter("material solid capacity",
                        material.solid.capacity,
                        "capacity of the solid phase");
      prm.add_parameter("material solid conductivity",
                        material.solid.conductivity,
                        "conductivity of the solid phase");
      prm.add_parameter("material solid density",
                        material.solid.density,
                        "density of the solid phase");
      prm.add_parameter("material solidus temperature",
                        material.solidus_temperature,
                        "Solidus temperature");
      prm.add_parameter("material liquidus temperature",
                        material.liquidus_temperature,
                        "Liquidus temperature");
    }
    prm.leave_subsection();
    /*
     *   paraview
     */
    prm.enter_subsection("paraview");
    {
      prm.add_parameter("paraview do output",
                        paraview.do_output,
                        "boolean for producing paraview output files");
      prm.add_parameter("paraview filename",
                        paraview.filename,
                        "Sets the base name for the paraview file output.");
      prm.add_parameter("paraview directory",
                        paraview.directory,
                        "Sets the base directory for the paraview file output.");
      prm.add_parameter("paraview write frequency",
                        paraview.write_frequency,
                        "every n timestep that should be written");
      prm.add_parameter("paraview do initial state",
                        paraview.do_initial_state,
                        "boolean for writing the initial state into the paraview output file");
      prm.add_parameter("paraview print levelset",
                        paraview.print_levelset,
                        "boolean for writing the levelset variable into the paraview output file");
      prm.add_parameter(
        "paraview print normal vector",
        paraview.print_normal_vector,
        "boolean for writing the computed normalvector into the paraview output file");
      prm.add_parameter("paraview print curvature",
                        paraview.print_curvature,
                        "boolean for writing the computed curvature into the paraview output file");
      prm.add_parameter("paraview print advection",
                        paraview.print_advection,
                        "boolean for writing the computed advection into the paraview output file");
      prm.add_parameter("paraview print exactsolution",
                        paraview.print_exactsolution,
                        "boolean for writing the exact solution into the paraview output file");
      prm.add_parameter("paraview print boundary id",
                        paraview.print_boundary_id,
                        "boolean for printing a vtk-file with the boundary id");
      prm.add_parameter("paraview n digits timestep",
                        paraview.n_digits_timestep,
                        "number of digits for the frame number of the vtk-file.");
      prm.add_parameter("paraview n groups",
                        paraview.n_digits_timestep,
                        "number of parallel written vtk-files.");
    }
    prm.leave_subsection();

    /*
     *   output
     */
    prm.enter_subsection("output");
    {
      prm.add_parameter(
        "do walltime",
        output.do_walltime,
        "this flag enables the output of wall times (should be disabled if a test file is prepared)");
      prm.add_parameter(
        "do compute error",
        output.do_compute_error,
        "this flag enables the computation of the error compared to a given analytical solution.");
      prm.add_parameter("do compute volume output",
                        output.do_compute_volume_output,
                        "boolean for computing the phase volumes");
      prm.add_parameter("filename volume output",
                        output.filename_volume_output,
                        "Sets the base name for the volume fraction file output.");
    }
    prm.leave_subsection();
  }

  template struct Parameters<double>;
} // namespace MeltPoolDG
