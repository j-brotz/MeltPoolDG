#include <meltpooldg/interface/boundary_conditions.hpp>

#include <algorithm>

namespace MeltPoolDG
{
  template <int dim>
  BoundaryTypes
  BoundaryConditions<dim>::get_boundary_type(types::boundary_id id)
  {
    if (dirichlet_bc.get_data().find(id) != dirichlet_bc.get_data().end())
      return BoundaryTypes::dirichlet_bc;
    else if (neumann_bc.find(id) != neumann_bc.end())
      return BoundaryTypes::neumann_bc;
    else if (std::find(outflow.begin(), outflow.end(), id) != outflow.end())
      return BoundaryTypes::outflow;
    else if (std::find(no_slip_bc.begin(), no_slip_bc.end(), id) != no_slip_bc.end())
      return BoundaryTypes::no_slip_bc;
    else if (std::find(fix_pressure_constant.begin(), fix_pressure_constant.end(), id) !=
             fix_pressure_constant.end())
      return BoundaryTypes::fix_pressure_constant;
    else if (std::find(symmetry_bc.begin(), symmetry_bc.end(), id) != symmetry_bc.end())
      return BoundaryTypes::symmetry_bc;
    else if (open_boundary_bc.find(id) != open_boundary_bc.end())
      return BoundaryTypes::open_boundary_bc;
    else if (std::find(radiation_bc.begin(), radiation_bc.end(), id) != radiation_bc.end())
      return BoundaryTypes::radiation_bc;
    else if (std::find(convection_bc.begin(), convection_bc.end(), id) != convection_bc.end())
      return BoundaryTypes::convection_bc;
    else if (inflow_outflow_bc.find(id) != inflow_outflow_bc.end())
      return BoundaryTypes::inflow_outflow;
    else
      {
        AssertThrow(false, ExcMessage("for specified boundary_id: " + std::to_string(id)));
        return BoundaryTypes::undefined;
      }
  }

  template <int dim>
  void
  BoundaryConditions<dim>::set_time(const double time)
  {
    dirichlet_bc.set_time(time);

    for (auto &n : neumann_bc)
      n.second->set_time(time);

    for (auto &n : inflow_outflow_bc)
      n.second->set_time(time);
  }

  template struct BoundaryConditions<1>;
  template struct BoundaryConditions<2>;
  template struct BoundaryConditions<3>;
} // namespace MeltPoolDG
