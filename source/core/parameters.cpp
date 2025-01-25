#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <filesystem>
#include <fstream>

namespace MeltPoolDG
{

  template <typename number>
  void
  Parameters<number>::add_parameters(ParameterHandler &prm)
  {
    base.add_parameters(prm);
    time_stepping.add_parameters(prm);
    amr.add_parameters(prm);
    ls.add_parameters(prm);
    heat.add_parameters(prm);
    laser.add_parameters(prm);
    rte.add_parameters(prm);
    /*
     *   melt pool TODO: move
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

    flow.add_parameters(prm);
    evapor.add_parameters(prm);
    material.add_parameters(prm);
    output.add_parameters(prm);
    profiling.add_parameters(prm);
    restart.add_parameters(prm);
    cut_param.add_parameters(prm);
  }

  template <typename number>
  void
  Parameters<number>::post(const std::string &parameter_filename)
  {
    /************************************************************************************
     * set input-file-dependent default parameters
     ************************************************************************************/
    amr.post(base.global_refinements, restart.load >= 0);
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

  template struct Parameters<double>;
} // namespace MeltPoolDG
