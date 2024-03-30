/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

#include <meltpooldg/interface/exceptions.hpp>

#include <map>
#include <memory>

namespace MeltPoolDG
{
  using namespace dealii;

  enum class BoundaryTypes
  {
    dirichlet_bc,
    neumann_bc,
    outflow,
    inflow_outflow,
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
  struct DirichletBoundaryConditions
  {
  private:
    using Type = std::map<types::boundary_id, std::shared_ptr<Function<dim>>>;
    Type bc;

  public:
    void
    attach(const types::boundary_id id, const std::shared_ptr<Function<dim>> f)
    {
      AssertThrow(bc.count(id) == 0, ExcBCAlreadyAssigned("Dirichlet"));

      bc[id] = f;
    }

    const Type &
    get_data() const
    {
      return bc;
    }

    void
    set_time(const double time)
    {
      for (auto &b : bc)
        b.second->set_time(time);
    }
  };

  template <int dim>
  struct BoundaryConditions
  {
    DirichletBoundaryConditions<dim>                             dirichlet_bc;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;
    std::vector<types::boundary_id>                              outflow;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> inflow_outflow_bc;
    std::vector<types::boundary_id>                              no_slip_bc;
    std::vector<types::boundary_id>                              fix_pressure_constant;
    std::vector<types::boundary_id>                              symmetry_bc;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> open_boundary_bc;
    std::vector<types::boundary_id>                              radiation_bc;
    std::vector<types::boundary_id>                              convection_bc;

    BoundaryTypes
    get_boundary_type(types::boundary_id id);

    void
    set_time(const double time);
  };
} // namespace MeltPoolDG
