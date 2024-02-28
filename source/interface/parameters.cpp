#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <filesystem>
#include <fstream>

namespace MeltPoolDG
{
  template <typename number>
  void
  Parameters<number>::process_parameters_file(ParameterHandler  &prm,
                                              const std::string &parameter_filename)
  {
    AssertThrow(!parameters_read, ExcMessage("The parameters are already read once."));

    /************************************************************************************
     * read *.json or *.prm file
     ************************************************************************************/

    add_parameters(prm);

    check_for_file(parameter_filename);

    {
      std::ifstream file;
      file.open(parameter_filename);

      if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "json")
        prm.parse_input_from_json(file, true);
      else if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "prm")
        prm.parse_input(parameter_filename);
      else
        AssertThrow(false, ExcMessage("Parameterhandler cannot handle current file ending"));
    }

    /************************************************************************************
     * set input-file-dependent default parameters
     ************************************************************************************/
    base.post();

    /*
     *  set the min grid refinement level if not user-specified
     */
    if (amr.min_grid_refinement_level == 1)
      amr.min_grid_refinement_level = base.global_refinements;

    /*
     * calculate the profiling output frequency if a time step size
     */
    if (profiling.write_time_step_size > 0.0)
      {
        AssertThrow(
          profiling.time_type == TimeType::real ||
            profiling.write_time_step_size >= time_stepping.time_step_size,
          ExcMessage("The time step size for profiling must be equal or larger than the simulation "
                     "time step size."));
      }

    // enable profiling for verbosity level higher than 1
    if (base.verbosity_level >= 1)
      profiling.enable = true;
    /*
     * calculate the restart output frequency if a time step size
     */
    if (restart.write_time_step_size > 0.0)
      {
        AssertThrow(restart.time_type == TimeType::real ||
                      restart.write_time_step_size >= time_stepping.time_step_size,
                    ExcMessage(
                      "The time step size for restart must be equal or larger than the simulation "
                      "time step size."));
      }
    else
      {
        restart.time_type            = TimeType::real;
        restart.write_time_step_size = 3600; // 1 hour
      }
    /*
     * set default value of restart prefix
     */
    namespace fs = std::filesystem;
    if (restart.directory == "")
      restart.directory = fs::path(fs::current_path()) / fs::path(output.directory);

    // modify the value of prefix for easy internal access
    restart.prefix = fs::path(restart.directory) / fs::path(restart.prefix);
    /*
     * do not allow initial refinement cycles in case of restart load
     */
    if (restart.load >= 0)
      amr.n_initial_refinement_cycles = 0;

    recoil.post(material);
    heat.post(base.degree, base.verbosity_level, material);
    laser.post(base.dimension,
               heat.use_volume_specific_thermal_capacity_for_phase_interpolation,
               material);
    ls.post(reinit.max_n_steps);
    advec_diff.post();
    reinit.post();
    normal_vec.post();
    curv.post();
    surface_tension.post(material);
    output.post(time_stepping.time_step_size, parameter_filename);

    /************************************************************************************
     * check input parameters for validity
     ************************************************************************************/
    check_input_parameters();

    parameters_read = true;
  }

  template <typename number>
  void
  Parameters<number>::check_input_parameters() const
  {
    base.check_input_parameters(ls.n_subdivisions);
    heat.check_input_parameters(base.do_simplex, ls.n_subdivisions);
    laser.check_input_parameters();
    ls.check_input_parameters(base.degree);
    evapor.check_input_parameters(ls.n_subdivisions);
    surface_tension.check_input_parameters(curv.enable);

    AssertThrow((advec_diff.conv_stab.type == ConvectionStabilizationType::SUPG &&
                 advec_diff.linear_solver.do_matrix_free &&
                 advec_diff.implementation == "meltpooldg") ||
                  advec_diff.conv_stab.type == ConvectionStabilizationType::none,
                ExcNotImplemented());
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
    base.add_parameters(prm);
    /*
     *   time stepping
     */
    time_stepping.add_parameters(prm);
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
      prm.enter_subsection("convection stabilization");
      {
        prm.add_parameter("type",
                          advec_diff.conv_stab.type,
                          "Defines the type for convection stabilization.");

        prm.add_parameter(
          "coefficient",
          advec_diff.conv_stab.coefficient,
          "Defines the stabilization coefficient for convection. (default velocity-dependent).");
      }
      prm.leave_subsection();
      prm.add_parameter("advec diff diffusivity",
                        advec_diff.diffusivity,
                        "Defines the diffusivity for the advection diffusion equation ");
      prm.add_parameter("advec diff time integration scheme",
                        advec_diff.time_integration_scheme,
                        "Determines the time integration scheme.",
                        Patterns::Selection("explicit_euler|implicit_euler|crank_nicolson|bdf_2"));
      prm.add_parameter(
        "advec diff implementation",
        advec_diff.implementation,
        "Choose the corresponding implementation of the advection diffusion operation.",
        Patterns::Selection("meltpooldg|adaflo"));
      advec_diff.predictor.add_parameters(prm);
      advec_diff.linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();

    /*
     *   levelset
     */
    ls.add_parameters(prm);
    /*
     *   reinitialization
     */
    reinit.add_parameters(prm);
    /*
     *   normal vector
     */
    prm.enter_subsection("normal vector");
    {
      prm.add_parameter(
        "normal vec damping scale factor",
        normal_vec.damping_scale_factor,
        "normal vector computation: damping = cell_size * normal_vec_damping_scale_factor");
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
      prm.add_parameter(
        "narrow band threshold",
        normal_vec.narrow_band_threshold,
        "If >> normal vec do narrow band << is set to true this parameter determines the level set "
        "treshold for the narrow band.");
      normal_vec.predictor.add_parameters(prm);
      normal_vec.linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();
    /*
     *   curvature
     */
    prm.enter_subsection("curvature");
    {
      prm.add_parameter(
        "enable",
        curv.enable,
        "Set this parameter to true if curvature should be computed. This is required in case of "
        "surface tension.");
      prm.add_parameter("curv damping scale factor",
                        curv.damping_scale_factor,
                        "curvature computation: damping = cell_size * curv_damping_scale_factor");
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
      prm.add_parameter(
        "narrow band threshold",
        curv.narrow_band_threshold,
        "If >> curv do narrow band << is set to true this parameter determines the level set "
        "treshold for the narrow band.");
      curv.predictor.add_parameters(prm);
      curv.linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();
    /*
     *   heat
     */
    heat.add_parameters(prm);
    /*
     *   laser
     */
    laser.add_parameters(prm);
    /*
     *   radiative transfer equaton (RTE)
     */
    rte.add_parameters(prm);
    /*
     * recoil pressure
     */
    recoil.add_parameters(prm);
    /*
     *   melt pool
     */
    prm.enter_subsection("melt pool");
    {
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
    }
    prm.leave_subsection();
    /*
     * surface tension
     */
    surface_tension.add_parameters(prm);
    /*
     * Darcy Damping
     */
    darcy.add_parameters(prm);
    /*
     *  evaporation
     */
    evapor.add_parameters(prm);
    /*
     *  material
     */
    material.add_parameters(prm);
    /*
     *   output
     */
    output.add_parameters(prm);
    /*
     *   profiling
     */
    prm.enter_subsection("profiling");
    {
      prm.add_parameter(
        "enable",
        profiling.enable,
        "Set this parameter to true if profiling should be enabled. It will be automatically"
        "enabled for verbosity level >=1.");
      prm.add_parameter("write time step size",
                        profiling.write_time_step_size,
                        "Write profiling output every given time step size. If this parameter is "
                        "set, the specified parameter for write frequency is overwritten.");
      prm.add_parameter("time type",
                        restart.time_type,
                        "Choose the type of time measure to write profiling information.");
    }
    prm.leave_subsection();
    /*
     *   restart
     */
    prm.enter_subsection("restart");
    {
      prm.add_parameter(
        "save",
        restart.save,
        "Set this parameter to any number >= 0 to specify how many restart files should be kept. "
        "-1 means no restart save.");
      prm.add_parameter(
        "load",
        restart.load,
        "Set this parameter to any number >= 0 to specify which restart file should be loaded. "
        "-1 means no restart load.");
      prm.add_parameter("write time step size",
                        restart.write_time_step_size,
                        "Write restart output every given time step size. If this parameter is "
                        "set, the specified parameter for write frequency is overwritten.");
      prm.add_parameter("time type",
                        restart.time_type,
                        "Choose the type of time measure to write restart.");
      prm.add_parameter("directory", restart.directory, "Write restart directory");
      prm.add_parameter("prefix", restart.prefix, "Write restart prefix");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  Parameters<number>::print_parameters(ParameterHandler &prm,
                                       std::ostream     &pcout,
                                       const bool        print_details)
  {
    // Set the written variable values to the ParameterHandler.
    //
    // @note: Here, potential rounding errors are introduced.
    add_parameters(prm);
    print_parameters_external(prm, pcout, print_details);
  }

  template struct Parameters<double>;
} // namespace MeltPoolDG
