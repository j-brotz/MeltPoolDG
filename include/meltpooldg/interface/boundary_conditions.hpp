/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <map>
#include <memory>
#include <string>

namespace MeltPoolDG
{
  using namespace dealii;

  enum class BoundaryTypes
  {
    dirichlet_bc,
    neumann_bc,
    outflow,
    no_slip_bc,
    fix_pressure_constant,
    symmetry_bc,
    periodic_bc,
    open_boundary_bc,
    radiation_bc,
    convection_bc,
    undefined
  };

  template <int dim>
  struct BoundaryConditions
  {
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;
    std::vector<types::boundary_id>                              outflow;
    std::vector<types::boundary_id>                              no_slip_bc;
    std::vector<types::boundary_id>                              fix_pressure_constant;
    std::vector<types::boundary_id>                              symmetry_bc;
    std::vector<types::boundary_id>                              open_boundary_bc;
    std::vector<types::boundary_id>                              radiation_bc;
    std::vector<types::boundary_id>                              convection_bc;

    BoundaryTypes
    get_boundary_type(types::boundary_id id);
  };
} // namespace MeltPoolDG
