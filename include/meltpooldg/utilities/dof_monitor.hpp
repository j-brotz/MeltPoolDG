#pragma once

#include <deal.II/base/convergence_table.h>

namespace MeltPoolDG
{
  class DoFMonitor
  {
  private:
    struct DoFStatistics
    {
      DoFStatistics(const unsigned int n_dofs = 0)
        : n_calls(1)
        , dofs_accumulated(n_dofs)
        , dofs_min(n_dofs)
        , dofs_max(n_dofs)
      {}

      unsigned int n_calls;
      unsigned int dofs_accumulated;
      unsigned int dofs_min;
      unsigned int dofs_max;
    };

  public:
    static void
    add_n_dofs(const std::string label, const unsigned int n_dofs)
    {
      const auto ptr = stat_dofs.find(label);

      if (ptr == stat_dofs.end())
        {
          stat_dofs[label] = DoFStatistics(n_dofs);
        }
      else
        {
          ptr->second.n_calls += 1;
          ptr->second.dofs_accumulated += n_dofs;
          ptr->second.dofs_min = std::min(ptr->second.dofs_min, n_dofs);
          ptr->second.dofs_max = std::max(ptr->second.dofs_max, n_dofs);
        }
    }

    template <typename StreamType>
    static void
    print(StreamType &ss)
    {
      ConvergenceTable table;

      for (const auto &entry : stat_dofs)
        {
          table.add_value("label", entry.first);
          table.add_value("n_calls", entry.second.n_calls);
          table.add_value("dofs_avg",
                          static_cast<double>(entry.second.dofs_accumulated) /
                            entry.second.n_calls);
          table.add_value("dofs_min", entry.second.dofs_min);
          table.add_value("dofs_max", entry.second.dofs_max);
        }

      if (ss.is_active())
        table.write_text(ss.get_stream(), TableHandler::TextOutputFormat::org_mode_table);
    }


  private:
    inline static std::map<std::string, DoFStatistics> stat_dofs;
  };
} // namespace MeltPoolDG
