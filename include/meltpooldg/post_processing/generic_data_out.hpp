#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>

#include <string>

using namespace dealii;

namespace MeltPoolDG
{
  template <int dim>
  class GenericDataOut
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    GenericDataOut(const Mapping<dim>            &mapping,
                   const double                   current_time,
                   const std::vector<std::string> req_var = {"all"});

    void
    add_data_vector(const DoFHandler<dim>          &dof_handler,
                    const VectorType               &data,
                    const std::vector<std::string> &names,
                    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
                      &data_component_interpretation =
                        std::vector<DataComponentInterpretation::DataComponentInterpretation>(),
                    const bool force_output = false);

    void
    add_data_vector(const DoFHandler<dim> &dof_handler,
                    const VectorType      &data,
                    const std::string     &name,
                    const bool             force_output = false);

    std::vector<std::tuple<const DoFHandler<dim> *,
                           const VectorType *,
                           const std::vector<std::string>,
                           std::vector<DataComponentInterpretation::DataComponentInterpretation>>>
      entries;

    const VectorType &
    get_vector(const std::string &name) const;

    const DoFHandler<dim> &
    get_dof_handler(const std::string &name) const;

    const Mapping<dim> &
    get_mapping() const;

    const double &
    get_time() const;

    bool
    is_requested(const std::string &var) const;

    std::vector<unsigned int>
    get_indices_data_request(const std::vector<std::string> req_var) const;

  private:
    std::map<std::string, unsigned int> entry_id;
    const Mapping<dim>                 &mapping;
    double                              current_time;
    const std::vector<std::string>      req_vars;
    mutable std::map<std::string, bool> req_vars_info;
    const bool                          req_all = false;
  };
} // namespace MeltPoolDG
