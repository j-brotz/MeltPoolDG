/* ---------------------------------------------------------------------
 *
 * Author: Nils Much, Magdalena Schreter, TUM, March 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  struct PeriodicBoundaryConditions
  {
  private:
    std::vector<
      std::tuple<types::boundary_id /*inflow*/, types::boundary_id /*outflow*/, int /*direction*/>>
      periodic_bc;

  public:
    void
    attach_boundary_condition(const types::boundary_id id_in,
                              const types::boundary_id id_out,
                              const int                direction);

    const std::vector<std::tuple<types::boundary_id, types::boundary_id, int>> &
    get_periodic_bc();
  };
} // namespace MeltPoolDG
