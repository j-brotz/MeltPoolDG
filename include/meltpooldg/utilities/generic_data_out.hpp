#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>

using namespace dealii;

namespace MeltPoolDG
{
  template <int dim>
  class GenericDataOut
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    GenericDataOut() = default;

    void
    add_data_vector(const DoFHandler<dim> &         dof_handler,
                    const VectorType &              data,
                    const std::vector<std::string> &names,
                    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
                      &data_component_interpretation =
                        std::vector<DataComponentInterpretation::DataComponentInterpretation>());

    void
    add_data_vector(const DoFHandler<dim> &dof_handler,
                    const VectorType &     data,
                    const std::string &    name);

    std::vector<std::tuple<const DoFHandler<dim> *,
                           const VectorType *,
                           const std::vector<std::string>,
                           std::vector<DataComponentInterpretation::DataComponentInterpretation>>>
      entries;
  };
} // namespace MeltPoolDG
