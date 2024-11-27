#include <meltpooldg/core/parameters_base.hpp>

namespace MeltPoolDG
{

  void
  add_and_parse_parameters(const std::string                             &parameter_file,
                           const std::function<void(ParameterHandler &)> &add_parameters,
                           const bool                                     enable_print,
                           const bool                                     print_details)
  {
    ParameterHandler prm;
    add_parameters(prm);

    std::ifstream file;
    file.open(parameter_file);

    if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "json")
      prm.parse_input_from_json(file, true);
    else if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "prm")
      prm.parse_input(parameter_file);
    else
      AssertThrow(false, ExcMessage("ParameterHandler cannot handle current file ending"));

    if (enable_print)
      prm.print_parameters(std::cout,
                           print_details ? ParameterHandler::OutputStyle::JSON |
                                             ParameterHandler::OutputStyle::KeepDeclarationOrder :
                                           ParameterHandler::OutputStyle::ShortJSON);
  }


  void
  ParametersBase::process_parameters_file(ParameterHandler  &prm,
                                          const std::string &parameter_filename)
  {
    AssertThrow(!parameters_read, ExcMessage("The parameters are already read once."));

    // Add parameters to the handler
    add_parameters(prm);

    // Ensure the parameter file exists
    check_for_file(parameter_filename);

    // Open and parse the parameter file
    std::ifstream file(parameter_filename);
    if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "json")
      {
        prm.parse_input_from_json(file, true);
      }
    else if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "prm")
      {
        prm.parse_input(parameter_filename);
      }
    else
      {
        AssertThrow(false, ExcMessage("ParameterHandler cannot handle current file extension."));
      }

    // Post-processing
    post(parameter_filename);

    parameters_read = true;
  }

  void
  ParametersBase::print_parameters(ParameterHandler &prm,
                                   std::ostream     &pcout,
                                   const bool        print_details)
  {
    add_parameters(prm); // Re-add parameters to ensure consistency

    prm.print_parameters(pcout,
                         print_details ? ParameterHandler::OutputStyle::JSON |
                                           ParameterHandler::OutputStyle::KeepDeclarationOrder :
                                         ParameterHandler::OutputStyle::ShortJSON);
  }

  void
  ParametersBase::check_for_file(const std::string &parameter_filename) const
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
} // namespace MeltPoolDG
