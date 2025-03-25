#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>

#include <string>

namespace MeltPoolDG
{
  template <int dim, typename number>
  class GenericDataOut
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    GenericDataOut(const dealii::Mapping<dim>    &mapping,
                   const number                   current_time,
                   const std::vector<std::string> req_var = {"all"});

    void
    add_data_vector(
      const dealii::DoFHandler<dim>  &dof_handler,
      const VectorType               &data,
      const std::vector<std::string> &names,
      const std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        &data_component_interpretation =
          std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>(),
      const bool force_output = false);

    void
    add_data_vector(const dealii::DoFHandler<dim> &dof_handler,
                    const VectorType              &data,
                    const std::string             &name,
                    const bool                     force_output = false);

    std::vector<
      std::tuple<const dealii::DoFHandler<dim> *,
                 const VectorType *,
                 const std::vector<std::string>,
                 std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>>>
      entries;

    const VectorType &
    get_vector(const std::string &name) const;

    const dealii::DoFHandler<dim> &
    get_dof_handler(const std::string &name) const;

    const dealii::Mapping<dim> &
    get_mapping() const;

    const number &
    get_time() const;

    bool
    is_requested(const std::string &var) const;

    std::vector<unsigned int>
    get_indices_data_request(const std::vector<std::string> req_var) const;

  private:
    std::map<std::string, unsigned int> entry_id;
    const dealii::Mapping<dim>         &mapping;
    number                              current_time;
    const std::vector<std::string>      req_vars;
    mutable std::map<std::string, bool> req_vars_info;
    const bool                          req_all = false;
  };
} // namespace MeltPoolDG
