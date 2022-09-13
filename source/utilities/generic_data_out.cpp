#include <meltpooldg/utilities/generic_data_out.hpp>

namespace MeltPoolDG
{
  template <int dim>
  GenericDataOut<dim>::GenericDataOut(const Mapping<dim> &mapping, const double current_time)
    : mapping(mapping)
    , current_time(current_time)
  {}

  template <int dim>
  void
  GenericDataOut<dim>::add_data_vector(
    const DoFHandler<dim> &         dof_handler,
    const VectorType &              data,
    const std::vector<std::string> &names,
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      &data_component_interpretation)
  {
    entries.emplace_back(&dof_handler, &data, names, data_component_interpretation);
    entry_id[names[0]] = entries.size() - 1;
  }

  template <int dim>
  void
  GenericDataOut<dim>::add_data_vector(const DoFHandler<dim> &dof_handler,
                                       const VectorType &     data,
                                       const std::string &    name)
  {
    entries.emplace_back(&dof_handler,
                         &data,
                         std::vector<std::string>{name},
                         std::vector<DataComponentInterpretation::DataComponentInterpretation>{
                           DataComponentInterpretation::component_is_scalar});
    entry_id[name] = entries.size() - 1;
  }

  template <int dim>
  const typename GenericDataOut<dim>::VectorType &
  GenericDataOut<dim>::get_vector(const std::string &name) const
  {
    if (entry_id.find(name) == entry_id.end())
      {
        std::ostringstream exc_message;
        exc_message << "Your requested output variable >>> " + name +
                         " <<< cannot be found in the entries"
                         " of GenericDataOut. The available values are: "
                    << std::endl;
        for (auto &[key, value] : entry_id)
          exc_message << "* " << key << std::endl;

        AssertThrow(false, ExcMessage(exc_message.str()));
      }

    return *std::get<1>(entries[entry_id.at(name)]);
  }

  template <int dim>
  const DoFHandler<dim> &
  GenericDataOut<dim>::get_dof_handler(const std::string &name) const
  {
    if (entry_id.find(name) == entry_id.end())
      {
        std::ostringstream exc_message;
        exc_message << "Your requested output variable >>> " + name +
                         " <<< cannot be found in the entries"
                         " of GenericDataOut. The available values are: "
                    << std::endl;
        for (auto &[key, value] : entry_id)
          exc_message << "* " << key << std::endl;

        AssertThrow(false, ExcMessage(exc_message.str()));
      }

    return *std::get<0>(entries[entry_id.at(name)]);
  }

  template <int dim>
  const Mapping<dim> &
  GenericDataOut<dim>::get_mapping() const
  {
    return mapping;
  }

  template <int dim>
  const double &
  GenericDataOut<dim>::get_time() const
  {
    return current_time;
  }

  template <int dim>
  std::vector<unsigned int>
  GenericDataOut<dim>::get_indices_data_request(const std::vector<std::string> req_var) const
  {
    AssertThrow((req_var.size() == 1 && req_var[0] == "all") ||
                  (std::find(req_var.begin(), req_var.end(), "all") == req_var.end()),
                ExcMessage(
                  "The requested output variables are ambiguous. Please specify either 'all' or "
                  "a list of specific output variables separated by a comma, e.g. 'var1,var2'."));

    // collect names of variables
    std::vector<std::string> names(entries.size());

    // collect indices of requested variables
    std::vector<unsigned int> req_idx;

    for (const auto &[name, i] : entry_id)
      {
        names.emplace_back(name);

        if (req_var[0] == "all" || std::find(req_var.begin(), req_var.end(), name) != req_var.end())
          req_idx.emplace_back(i);
      }

    std::ostringstream message;
    message << "One of your requested output variables could not be read. Please make sure "
            << "that the spelling is correct. Either choose 'all' or a comma separated list "
            << "of the following names: " << std::endl;

    for (const auto &n : names)
      if (!n.empty())
        message << "  * " << n << std::endl;

    AssertThrow(req_idx.size() == req_var.size() || req_var[0] == "all", ExcMessage(message.str()));

    AssertThrow(
      req_idx.size() > 0,
      ExcMessage("Your requested output variables could not be read. In case you don't want "
                 "to produce output set 'paraview do output' to false. Otherwise make sure that you"
                 " specify your variables in a comma separated list, e.g. 'var1,var2'."));

    return req_idx;
  }

  template class GenericDataOut<1>;
  template class GenericDataOut<2>;
  template class GenericDataOut<3>;
} // namespace MeltPoolDG
