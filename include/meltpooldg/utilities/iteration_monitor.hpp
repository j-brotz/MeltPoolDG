#pragma once

#include <deal.II/base/convergence_table.h>

namespace MeltPoolDG
{
  class IterationMonitor
  {
  private:
    struct LinearIterationStatistics
    {
      LinearIterationStatistics(const unsigned int n_iterations = 0)
        : n_calls(1)
        , iterations_accumulated(n_iterations)
        , iterations_min(n_iterations)
        , iterations_max(n_iterations)
      {}

      unsigned int n_calls;
      unsigned int iterations_accumulated;
      unsigned int iterations_min;
      unsigned int iterations_max;
    };

  public:
    static void
    add_linear_iterations(const std::string label, const unsigned int n_iterations)
    {
      const auto ptr = stat_linear.find(label);

      if (ptr == stat_linear.end())
        {
          stat_linear[label] = LinearIterationStatistics(n_iterations);
        }
      else
        {
          ptr->second.n_calls += 1;
          ptr->second.iterations_accumulated += n_iterations;
          ptr->second.iterations_min = std::min(ptr->second.iterations_min, n_iterations);
          ptr->second.iterations_max = std::max(ptr->second.iterations_max, n_iterations);
        }
    }

    template <typename StreamType>
    static void
    print(StreamType &ss)
    {
      ConvergenceTable table;

      for (const auto &entry : stat_linear)
        {
          table.add_value("label", entry.first);
          table.add_value("n_calls", entry.second.n_calls);
          table.add_value("iterations_avg",
                          static_cast<double>(entry.second.iterations_accumulated) /
                            entry.second.n_calls);
          table.add_value("iterations_min", entry.second.iterations_min);
          table.add_value("iterations_max", entry.second.iterations_max);
        }

      if (ss.is_active())
        table.write_text(ss.get_stream(), TableHandler::TextOutputFormat::org_mode_table);
    }


  private:
    inline static std::map<std::string, LinearIterationStatistics> stat_linear;
  };
} // namespace MeltPoolDG
