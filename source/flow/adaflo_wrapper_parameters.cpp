#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>

namespace MeltPoolDG::Flow
{
#ifdef MELT_POOL_DG_WITH_ADAFLO
  void
  AdafloWrapperParameters::parse_parameters(const std::string &parameter_filename)
  {
    ParameterHandler prm_adaflo;

    // declare parameters
    {
      prm_adaflo.enter_subsection("flow");
      prm_adaflo.enter_subsection("adaflo");
      params.declare_parameters(prm_adaflo);
      prm_adaflo.leave_subsection();
      prm_adaflo.leave_subsection();
    }

    // parse parameters
    {
      std::ifstream file;
      file.open(parameter_filename);

      if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "json")
        prm_adaflo.parse_input_from_json(file, true);
      else if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "prm")
        prm_adaflo.parse_input(parameter_filename);
      else
        AssertThrow(false, ExcMessage("Parameterhandler cannot handle current file ending"));
    }

    // read parsed parameters
    {
      prm_adaflo.enter_subsection("flow");
      prm_adaflo.enter_subsection("adaflo");
      params.parse_parameters(parameter_filename, prm_adaflo);
      prm_adaflo.leave_subsection();
      prm_adaflo.leave_subsection();
    }
  }

  const FlowParameters &
  AdafloWrapperParameters::get_parameters() const
  {
    return this->params;
  }
#endif

} // namespace MeltPoolDG::Flow
