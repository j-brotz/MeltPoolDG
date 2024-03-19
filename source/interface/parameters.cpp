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
    /*
     *  set the min grid refinement level if not user-specified
     */
    if (amr.min_grid_refinement_level == 1)
      amr.min_grid_refinement_level = base.global_refinements;
    /*
     * do not allow initial refinement cycles in case of restart load
     */
    if (restart.load >= 0)
      amr.n_initial_refinement_cycles = 0;

    heat.post(base.fe, base.verbosity_level);
    laser.post(base.dimension,
               heat.use_volume_specific_thermal_capacity_for_phase_interpolation,
               material);
    ls.post(base.fe);
    evapor.post(material, heat.use_volume_specific_thermal_capacity_for_phase_interpolation);
    flow.post(material);
    output.post(time_stepping.time_step_size, parameter_filename);
    profiling.post(base.verbosity_level);
    restart.post(output.directory);

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
    base.check_input_parameters(ls.get_n_subdivisions());
    heat.check_input_parameters(base.fe);
    laser.check_input_parameters();
    ls.check_input_parameters(base.fe);
    evapor.check_input_parameters(material, ls.get_n_subdivisions());
    flow.check_input_parameters(ls.curv.enable);
    profiling.check_input_parameters(time_stepping.time_step_size);
    restart.check_input_parameters(time_stepping.time_step_size);
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
     *   levelset
     */
    ls.add_parameters(prm);
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
     * flow
     */
    flow.add_parameters(prm);
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
    profiling.add_parameters(prm);
    /*
     *   restart
     */
    restart.add_parameters(prm);
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
