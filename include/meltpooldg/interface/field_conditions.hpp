#pragma once
#include <deal.II/base/function.h>

#include <memory>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  struct FieldConditions
  {
    std::shared_ptr<Function<dim>> initial_field;
    std::shared_ptr<Function<dim>> source_field;
    std::shared_ptr<Function<dim>> advection_field;
    std::shared_ptr<Function<dim>> velocity_field;
    std::shared_ptr<Function<dim>> exact_solution_field;
  };
} // namespace MeltPoolDG
