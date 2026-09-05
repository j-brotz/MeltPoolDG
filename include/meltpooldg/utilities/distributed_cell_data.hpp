#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <functional>
#include <optional>
#include <vector>

namespace MeltPoolDG::Utilities
{
  template <int dim, typename cell_data_type>
  class DistributedCellData
  {
    using ActiveCellIterator = typename dealii::Triangulation<dim>::active_cell_iterator;

  public:
    explicit DistributedCellData(const dealii::Triangulation<dim> &tria_in)
      : tria(tria_in)
      , cell_data(tria.n_active_cells())
    {}

    cell_data_type &
    operator[](const dealii::types::global_cell_index active_cell_index)
    {
      AssertIndexRange(active_cell_index, tria.n_active_cells());
      return cell_data[active_cell_index];
    }

    const cell_data_type &
    operator[](const dealii::types::global_cell_index active_cell_index) const
    {
      AssertIndexRange(active_cell_index, tria.n_active_cells());
      return cell_data[active_cell_index];
    }

    cell_data_type *
    data()
    {
      return cell_data.data();
    }

    typename std::vector<cell_data_type>::iterator
    begin()
    {
      return cell_data.begin();
    }

    typename std::vector<cell_data_type>::iterator
    end()
    {
      return cell_data.end();
    }

    typename std::vector<cell_data_type>::const_iterator
    cbegin() const
    {
      return cell_data.cbegin();
    }

    typename std::vector<cell_data_type>::const_iterator
    cend() const
    {
      return cell_data.cend();
    }

    void
    update_ghost_values()
    {
      std::function<std::optional<cell_data_type>(const ActiveCellIterator &)> pack =
        [&](const ActiveCellIterator &cell) -> std::optional<cell_data_type> {
        return cell_data[cell->active_cell_index()];
      };

      std::function<void(const ActiveCellIterator &, const cell_data_type &)> unpack =
        [&](const ActiveCellIterator &cell, const cell_data_type &value) {
          cell_data[cell->active_cell_index()] = value;
        };

      dealii::GridTools::exchange_cell_data_to_ghosts(tria, pack, unpack);
    }

  private:
    const dealii::Triangulation<dim> &tria;
    std::vector<cell_data_type>       cell_data;
  };
} // namespace MeltPoolDG::Utilities
