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
  const GenericDataOut<dim>::VectorType &
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

  template class GenericDataOut<1>;
  template class GenericDataOut<2>;
  template class GenericDataOut<3>;
} // namespace MeltPoolDG
