#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <vector>

namespace MeltPoolDG
{
  template <int dim>
  struct PeriodicBoundaryConditions
  {
  private:
    using Type = std::vector<std::tuple<dealii::types::boundary_id /*inflow*/,
                                        dealii::types::boundary_id /*outflow*/,
                                        int /*direction*/>>;

    Type periodic_bc;

  public:
    void
    attach_boundary_condition(const dealii::types::boundary_id id_in,
                              const dealii::types::boundary_id id_out,
                              const int                        direction);

    const Type &
    get_data() const;
  };
} // namespace MeltPoolDG
