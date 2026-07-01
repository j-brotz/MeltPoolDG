#include <meltpooldg/core/parameters_base.hpp>

namespace MeltPoolDG
{

  void
  add_and_parse_parameters(const std::string                                     &parameter_file,
                           const std::function<void(dealii::ParameterHandler &)> &add_parameters,
                           const bool                                            &enable_print,
                           const bool                                             print_details,
                           const bool                                             skip_undefined,
                           const std::string                                     &subsection)
  {
    dealii::ParameterHandler prm;

    if (!subsection.empty())
      prm.enter_subsection(subsection);

    add_parameters(prm);

    if (!subsection.empty())
      prm.leave_subsection();

    std::ifstream file;
    file.open(parameter_file);

    if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "json")
      prm.parse_input_from_json(file, skip_undefined /*skip_undefined*/);
    else if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "prm")
      prm.parse_input(parameter_file, "", skip_undefined);
    else
      AssertThrow(false, dealii::ExcMessage("ParameterHandler cannot handle current file ending"));

    if (enable_print)
      prm.print_parameters(std::cout,
                           print_details ?
                             dealii::ParameterHandler::OutputStyle::JSON |
                               dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder :
                             dealii::ParameterHandler::OutputStyle::ShortJSON);
  }


  void
  ParametersBase::process_parameters_file(dealii::ParameterHandler &prm,
                                          const std::string        &parameter_filename)
  {
    AssertThrow(!parameters_read, dealii::ExcMessage("The parameters are already read once."));

    // Add parameters to the handler
    add_parameters(prm);

    // Ensure the parameter file exists
    check_for_file(parameter_filename);

    // Open and parse the parameter file
    std::ifstream file(parameter_filename);
    if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "json")
      {
        prm.parse_input_from_json(file, true /*skip_undefined*/);
      }
    else if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "prm")
      {
        prm.parse_input(parameter_filename, "", true /*skip_undefined*/);
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage("ParameterHandler cannot handle current "
                                       "file extension."));
      }

    // Post-processing
    post(parameter_filename);

    parameters_read = true;
  }

  void
  ParametersBase::print_parameters(dealii::ParameterHandler &prm,
                                   std::ostream             &pcout,
                                   const bool                print_details)
  {
    add_parameters(prm); // Re-add parameters to ensure consistency

    prm.print_parameters(pcout,
                         print_details ?
                           dealii::ParameterHandler::OutputStyle::JSON |
                             dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder :
                           dealii::ParameterHandler::OutputStyle::ShortJSON);
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

        AssertThrow(false, dealii::ExcMessage(message.str().c_str()));
      }
  }
} // namespace MeltPoolDG
