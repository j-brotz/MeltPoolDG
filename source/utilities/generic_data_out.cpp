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
  }

  template class GenericDataOut<1>;
  template class GenericDataOut<2>;
  template class GenericDataOut<3>;
} // namespace MeltPoolDG
