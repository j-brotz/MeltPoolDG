#include <meltpooldg/interface/parameters.hpp>

#include <filesystem>

namespace MeltPoolDG
{
  template <typename number>
  void
  Parameters<number>::process_parameters_file(const std::string &parameter_filename)
  {
    AssertThrow(!parameters_read, ExcMessage("The parameters are already read once."));

    ParameterHandler prm;
    add_parameters(prm);

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
     * The level set problem for simplices can only be solved when no subdivision of the
     * finite element is undertaken.
     */
    AssertThrow((!base.do_simplex || ls.n_subdivisions == 1),
                ExcMessage(
                  "If you use a simplex mesh, n_subdivisions for the level set must be 1."));
    AssertThrow((ls.n_subdivisions == 1 || base.degree == 1),
                ExcMessage("If you use n_subdivisions for the level set, degree must be 1."));


    AssertThrow((ls.n_subdivisions == 1 ||
                 (evapor.formulation_evaporative_mass_flux_over_interface != "interface value" &&
                  evapor.formulation_evaporative_mass_flux_over_interface != "line integral")),
                ExcMessage(
                  "If you use the formulation of the evaporative mass flux over the interface "
                  "using the value at the interface or a line integral, n_subdivisions for the "
                  "level set must be 1."));

    switch (base.problem_name)
      {
        case ProblemType::advection_diffusion:
        case ProblemType::reinitialization:
        case ProblemType::heat_transfer:
          AssertThrow(
            ls.n_subdivisions == 1,
            ExcMessage(
              "n_subdivisions for the level set is not supported for your requested problem_type."));
          break;
        default:
          break;
      }
    /*
     * calculate the paraview write frequency if a time step for producing the output
     * is specified
     */
    if (paraview.write_time_step_size > 0.0)
      {
        AssertThrow(
          paraview.write_time_step_size >= time_stepping.time_step_size,
          ExcMessage(
            "The "
            "time step size for writing paraview files must be equal or larger than the simulation "
            "time step size."));
        paraview.write_frequency = paraview.write_time_step_size / time_stepping.time_step_size;
      }

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
    if (laser.analytical.max_temperature < material.boiling_temperature)
      laser.analytical.max_temperature = material.boiling_temperature + 500;
    /*
     *  check if level set assignment of gaseous/liquid phase is done correctly
     */
    if (evapor.ls_value_liquid == evapor.ls_value_gas)
      AssertThrow(
        false, ExcMessage("Parameterhandler: ls value liquid must not be equal to ls value gas."));


    if (heat.solidification)
      {
        material.inv_mushy_interval =
          1.0 / (material.liquidus_temperature -
                 material.solidus_temperature); //@todo: move to new material class
      }

    // create output directory and copy parameter file
    {
      namespace fs = std::filesystem;
      // check if the requested paraview directory exists and if not create the directory
      AssertThrow(!fs::exists(paraview.directory) || fs::is_directory(paraview.directory),
                  ExcMessage("You are trying to create a folder with the name <" +
                             std::string(paraview.directory) +
                             ">. However, a file with the same name already exists! "
                             "Possible solutions could be to either rename the output "
                             "folder in the parameter file or to rename/move the existing file."));

      if (!fs::exists(paraview.directory))
        fs::create_directory(paraview.directory);


      try
        {
          fs::copy(parameter_filename, paraview.directory, fs::copy_options::overwrite_existing);
        }
      catch (...)
        {
          // copy parameter file (workaround since overwrite_existing complains with certain
          // compilers)
          const auto path_orig = fs::path(parameter_filename);
          const auto path_dest =
            fs::path(paraview.directory) / fs::path(parameter_filename).filename();

          if (!fs::equivalent(path_orig, path_dest))
            {
              if (fs::exists(path_dest))
                fs::remove(path_dest);

              fs::copy(path_orig, path_dest, fs::copy_options::overwrite_existing);
            }
        }
    }

    parameters_read = true;
  }

  template <typename number>
  void
  Parameters<number>::print_parameters(std::ostream &pcout)
  {
    if (base.do_print_parameters)
      {
        ParameterHandler prm;
        add_parameters(prm);
        prm.print_parameters(pcout,
                             ParameterHandler::OutputStyle::Text |
                               ParameterHandler::OutputStyle::KeepDeclarationOrder);
      }
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
  Parameters<number>::add_parameters(ParameterHandler &prm)
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
      prm.add_parameter("problem name",
                        base.problem_name,
                        "Sets the base name for the problem that should be solved.");
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
                        "Set this parameter to true to list parameters in output");
      prm.add_parameter("do simplex", base.do_simplex, "Use simplices");
      prm.add_parameter("gravity", base.gravity, "Set the value for the gravity");
      prm.add_parameter(
        "verbosity level",
        base.verbosity_level,
        "Sets the maximum verbosity level of the console output. Set this parameter to 0 in case of test files.");
    }
    prm.leave_subsection();
    /*
     *   time stepping
     */
    prm.enter_subsection("time stepping");
    {
      prm.add_parameter("start time",
                        time_stepping.start_time,
                        "Defines the start time for the solution of the levelset problem");
      prm.add_parameter("end time",
                        time_stepping.end_time,
                        "Sets the end time for the solution of the levelset problem");
      prm.add_parameter("time step size",
                        time_stepping.time_step_size,
                        "Sets the step size for time stepping. For non-uniform "
                        "time stepping, this parameter determines the size of the first "
                        "time step.");
      prm.add_parameter("max n steps",
                        time_stepping.max_n_steps,
                        "Sets the maximum number of melt_pool steps");
    }
    prm.leave_subsection();
    /*
     *    adaptive meshing
     */
    prm.enter_subsection("adaptive meshing");
    {
      prm.add_parameter("do amr",
                        amr.do_amr,
                        "Set this parameter to true to activate adaptive meshing");
      prm.add_parameter("do not modify boundary cells",
                        amr.do_not_modify_boundary_cells,
                        "Set this parameter to true to not refine/coarsen along boundaries.");
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
      prm.add_parameter(
        "ls n subdivisions",
        ls.n_subdivisions,
        "Set the number of subdivisions for the finite element of the level set operation.");
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
      prm.add_parameter("reinit solver type",
                        reinit.solver.solver_type,
                        "Set this parameter for choosing a solver type.");
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
      prm.add_parameter(
        "normal vec verbosity level",
        normal_vec.verbosity_level,
        "Sets the maximum verbosity level of the console output. The maximum level with respect to the "
        " base value is decisive.");
      prm.add_parameter(
        "normal vec do narrow band",
        normal_vec.do_narrow_band,
        "Set this parameter to true to compute the normal vector only in the interfacial region.");
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
      prm.add_parameter(
        "curv verbosity level",
        curv.verbosity_level,
        "Sets the maximum verbosity level of the console output. The maximum level with respect to the "
        " base value is decisive.");
      prm.add_parameter(
        "curv do narrow band",
        curv.do_narrow_band,
        "Set this parameter to true to compute the curvature only in the interfacial region.");
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
      prm.add_parameter("heat velocity", heat.velocity, "Velocity.");
      prm.add_parameter("heat two phase", heat.two_phase, "Set this parameter for two phase flow.");
      prm.add_parameter("heat solidification",
                        heat.solidification,
                        "Set this parameter to true to consider solidification.");
      heat.delta_approximation_phase_weighted.add_parameters(prm);
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
      prm.add_parameter("laser scan speed", laser.scan_speed, "Scan speed of the laser.");
      prm.add_parameter(
        "laser do move",
        laser.do_move,
        "Set this parameter to true to move the laser in x-direction with the given parameter scan speed.");
      prm.add_parameter("laser heat source model",
                        laser.heat_source_model,
                        "Laser heat source model.");
      /*
       *   Gusarov
       */
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
      /*
       *   Gauss
       */
      prm.add_parameter(
        "laser impact type",
        laser.impact_type,
        "Laser impact model. volumetric: volumetric heat source | surface: surface heat source at two-phase interface.",
        Patterns::Selection("volumetric|interface"));
      prm.add_parameter("laser gauss laser beam radius",
                        laser.gauss.laser_beam_radius,
                        "Laser beam radius.");
      prm.add_parameter("laser gauss absorptivity gas",
                        laser.gauss.absorptivity_gas,
                        "Laser energy absorptivity of the gaseous part of the domain.");
      prm.add_parameter("laser gauss absorptivity liquid",
                        laser.gauss.absorptivity_liquid,
                        "Laser energy absorptivity of the liquid part of the domain.");
      laser.gauss.delta_approximation_phase_weighted.add_parameters(prm);
      /*
       *   Analytical temperature field
       */
      prm.add_parameter("laser analytical absorptivity liquid",
                        laser.analytical.absorptivity_liquid,
                        "Absorptivity of the liquid part of domain");
      prm.add_parameter("laser analytical absorptivity gas",
                        laser.analytical.absorptivity_gas,
                        "Absorptivity of the gaseous part of domain");
      prm.add_parameter("laser analytical ambient temperature",
                        laser.analytical.ambient_temperature,
                        "Ambient temperature in the inert gas.");
      prm.add_parameter(
        "laser analytical max temperature",
        laser.analytical.max_temperature,
        "Maximum temperature arising in the melt pool. If this temperature is lower than the boiling"
        " temperature, this value is corrected to correspond to the boiling temperature + 500 K.");
      prm.add_parameter("laser analytical temperature x to y ratio",
                        laser.analytical.temperature_x_to_y_ratio,
                        "This factor scales the analytical temperature field to be anisotropic.");
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
      recoil.delta_approximation_phase_weighted.add_parameters(prm);
    }
    prm.leave_subsection();
    /*
     *   melt pool
     */
    prm.enter_subsection("melt pool");
    {
      prm.add_parameter("mp melt pool center",
                        mp.melt_pool_center,
                        "Center coordinates of the melt pool ellipse/parabola. If no value is "
                        "provided it will be set equally to the laser center");
      prm.add_parameter(
        "mp set velocity to zero in solid",
        mp.solid.set_velocity_to_zero,
        "Set this parameter to true to constrain the flow velocity in the solid domain.");
      prm.add_parameter(
        "mp do not reinitialize in solid",
        mp.solid.do_not_reinitialize,
        "Set this parameter to true to forbid reinitialization of the level set field the solid domain.");
      prm.add_parameter(
        "mp solid fraction lower limit",
        mp.solid.solid_fraction_lower_limit,
        "Lower limit of the solid fraction for where the flow velocity / level set is "
        "set to zero if \"mp set velocity to zero in solid\" or \"mp set level set to zero in solid\" "
        "are enabled.",
        Patterns::Double(0.0, 1.0));
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
    }
    prm.leave_subsection();
    /*
     * surface tension
     */
    prm.enter_subsection("surface tension");
    {
      prm.add_parameter("surface tension coefficient",
                        surface_tension.surface_tension_coefficient,
                        "Constant coefficient for calculating surface tension");
      prm.add_parameter("temperature dependent surface tension coefficient",
                        surface_tension.temperature_dependent_surface_tension_coefficient,
                        "Temperature-dependent coefficient for calculating temperetaure-dependent "
                        "surface tension (Marangoni convection)");
      prm.add_parameter("reference temperature",
                        surface_tension.reference_temperature,
                        "Reference temperature for calculating surface tension");
      prm.add_parameter(
        "coefficient residual fraction",
        surface_tension.coefficient_residual_fraction,
        "Define the minimum fraction of the constant surface tension reference value "
        "that can be reached.");
      surface_tension.delta_approximation_phase_weighted.add_parameters(prm);
      prm.add_parameter(
        "zero surface tension in solid",
        surface_tension.zero_surface_tension_in_solid,
        "Set this parameter to true to only apply surface tension if the solid fraction is zero.");
    }
    prm.leave_subsection();
    /*
     * Darcy Damping
     */
    prm.enter_subsection("darcy damping");
    {
      prm.add_parameter("mushy zone morphology",
                        darcy.mushy_zone_morphology,
                        "Mushy zone morphology for Darcy damping");
      prm.add_parameter("avoid div zero constant",
                        darcy.avoid_div_zero_constant,
                        "This parameter exists to avoid division by zero in the "
                        "Kozeny–Carman equation for the Darcy damping force.");
      prm.add_parameter("formulation",
                        darcy.formulation,
                        "Set the formulation of the Darcy damping force.",
                        Patterns::Selection("explicit_formulation|implicit_formulation"));
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
      prm.add_parameter(
        "evapor evaporative mass flux",
        evapor.evaporative_mass_flux,
        "Prescribe a spatially and temporally constant mass flux due to evaporation (SI unit in kg/m²s).");
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
      prm.add_parameter("evapor interface value n iterations",
                        evapor.interface_value_n_iterations,
                        "Number of corrections of the point projection towards the interface.");
      prm.add_parameter("evapor line integral n subdivisions per side",
                        evapor.line_integral_n_subdivisions_per_side,
                        "Number of subdivisions per side to compute the points perpendicular to "
                        "the interface for the evaporative mass flux evaluation by "
                        "means of the line integral.");
      prm.add_parameter("evapor line integral n subdivisions MCA",
                        evapor.line_integral_n_subdivisions_MCA,
                        "Number of subdivisions for the marching cube algorithm within the "
                        "evaporative mass flux evaluation by means of the line integral.");
    }
    prm.leave_subsection();
    /*
     *  material
     */
    prm.enter_subsection("material");
    {
      material.add_parameters(prm);
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
      prm.add_parameter("paraview write time step size",
                        paraview.write_time_step_size,
                        "Write paraview output every given time step. If this parameter is "
                        "set, the paraview write frequency is overwritten.");
      prm.add_parameter("paraview do initial state",
                        paraview.do_initial_state,
                        "boolean for writing the initial state into the paraview output file");
      prm.add_parameter("paraview print boundary id",
                        paraview.print_boundary_id,
                        "boolean for printing a vtk-file with the boundary id");
      prm.add_parameter("paraview output subdomains",
                        paraview.output_subdomains,
                        "boolean for outputting the subdomain ranks");
      prm.add_parameter("paraview n digits timestep",
                        paraview.n_digits_timestep,
                        "number of digits for the frame number of the vtk-file.");
      prm.add_parameter("paraview n groups",
                        paraview.n_groups,
                        "number of parallel written vtk-files.");
      prm.add_parameter("paraview n patches",
                        paraview.n_patches,
                        "Control number of patches to enable high-order output.");
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
