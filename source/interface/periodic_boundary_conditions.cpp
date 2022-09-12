#include <meltpooldg/interface/periodic_boundary_conditions.hpp>

#include <tuple>

namespace MeltPoolDG
{
  template <int dim>
  void
  PeriodicBoundaryConditions<dim>::attach_boundary_condition(const types::boundary_id id_in,
                                                             const types::boundary_id id_out,
                                                             const int                direction)
  {
    AssertThrow(direction < dim, ExcMessage("Coordinate direction must be between 0 and the dim"));

    // check if requested tuple already exists
    for (const auto &bc : periodic_bc)
      {
        AssertThrow(!(std::get<0>(bc) == id_in && std::get<1>(bc) == id_in &&
                      std::get<2>(bc) == direction),
                    ExcMessage("The given periodic boundary condition < " + std::to_string(id_in) +
                               "," + std::to_string(id_out) + "," + std::to_string(direction) +
                               "> already exists. Make sure that periodic boundary conditions "
                               "must be unique."));
      }

    periodic_bc.emplace_back(id_in, id_out, direction);
  }

  template <int dim>
  const typename PeriodicBoundaryConditions<dim>::Type &
  PeriodicBoundaryConditions<dim>::get_data() const
  {
    return periodic_bc;
  }

  template struct PeriodicBoundaryConditions<1>;
  template struct PeriodicBoundaryConditions<2>;
  template struct PeriodicBoundaryConditions<3>;
} // namespace MeltPoolDG
