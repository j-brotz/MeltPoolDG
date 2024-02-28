#pragma once

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/table_handler.h>

#include <algorithm>
#include <map>
#include <string>

namespace MeltPoolDG
{
  class CellMonitor
  {
  private:
    struct CellStatistics
    {
      CellStatistics(const unsigned int n_cells       = 0,
                     const double       cell_size_min = 0,
                     const double       cell_size_max = 0)
        : n_calls(1)
        , cells_accumulated(n_cells)
        , cells_min(n_cells)
        , cells_max(n_cells)
        , cell_size_min(cell_size_min)
        , cell_size_max(cell_size_max)
      {}

      unsigned int n_calls;
      unsigned int cells_accumulated;
      unsigned int cells_min;
      unsigned int cells_max;

      double cell_size_min;
      double cell_size_max;
    };

  public:
    static void
    add_info(const std::string  label,
             const unsigned int n_cells,
             const double       min_cell_size,
             const double       max_cell_size)
    {
      const auto ptr = stat_cells.find(label);

      if (ptr == stat_cells.end())
        {
          stat_cells[label] = CellStatistics(n_cells, min_cell_size, max_cell_size);
        }
      else
        {
          ptr->second.n_calls += 1;
          ptr->second.cells_accumulated += n_cells;
          ptr->second.cells_min     = std::min(ptr->second.cells_min, n_cells);
          ptr->second.cells_max     = std::max(ptr->second.cells_max, n_cells);
          ptr->second.cell_size_min = min_cell_size;
          ptr->second.cell_size_max = max_cell_size;
        }
    }

    template <typename StreamType>
    static void
    print(StreamType &ss)
    {
      {
        dealii::ConvergenceTable table;

        for (const auto &entry : stat_cells)
          {
            table.add_value("label", entry.first);
            table.add_value("n_calls", entry.second.n_calls);
            table.add_value("n_cells_avg",
                            static_cast<double>(entry.second.cells_accumulated) /
                              entry.second.n_calls);
            table.add_value("n_cells_min", entry.second.cells_min);
            table.add_value("n_cells_max", entry.second.cells_max);
          }

        if (ss.is_active())
          table.write_text(ss.get_stream(), dealii::TableHandler::TextOutputFormat::org_mode_table);
      }

      {
        dealii::ConvergenceTable table;

        for (const auto &entry : stat_cells)
          {
            table.add_value("label", entry.first);
            table.add_value("cell_size_min", entry.second.cell_size_min);
            table.add_value("cell_size_max", entry.second.cell_size_max);
          }

        table.set_scientific("cell_size_min", 4);
        table.set_scientific("cell_size_max", 4);

        if (ss.is_active())
          table.write_text(ss.get_stream(), dealii::TableHandler::TextOutputFormat::org_mode_table);
      }
    }


  private:
    inline static std::map<std::string, CellStatistics> stat_cells;
  };
} // namespace MeltPoolDG
