#pragma once

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <meltpooldg/interface/finite_element_data.hpp>

#include <memory>

namespace MeltPoolDG::FiniteElementUtils
{
  using namespace dealii;

  template <int dim, unsigned int n_components = 1>
  void
  distribute_dofs(const FiniteElementData &fe_data, DoFHandler<dim> &dof_handler);

  template <int dim>
  std::shared_ptr<Mapping<dim>>
  create_mapping(const FiniteElementData &fe_data);

  template <int dim>
  Quadrature<dim>
  create_quadrature(const FiniteElementData &fe_data);
} // namespace MeltPoolDG::FiniteElementUtils