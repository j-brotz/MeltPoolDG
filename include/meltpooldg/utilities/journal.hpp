#pragma once
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>

namespace MeltPoolDG::Journal
{
  static constexpr int max_text_width = 100;

  inline void
  print_decoration_line(const ConditionalOStream &pcout)
  {
    pcout << "+" << std::string(max_text_width - 2, '-') << "+" << std::endl;
  }


  void
  print_line(const ConditionalOStream &pcout,
             const std::string &       text           = "",
             const std::string &       operation_name = "",
             const unsigned int        extra_size     = 0);
  inline void
  print_end(const ConditionalOStream &pcout)
  {
    print_decoration_line(pcout);
    print_line(pcout, " end of simulation");
    print_decoration_line(pcout);
  }

  inline void
  print_start(const ConditionalOStream &pcout)
  {
    print_decoration_line(pcout);
    print_line(pcout, std::string(48, ' ') + "MeltPoolDG");
    print_decoration_line(pcout);
  }

  void
  print_formatted_norm(const ConditionalOStream &pcout,
                       const double              norm_value,
                       const std::string &       norm_id,
                       const std::string &       operation_name,
                       const unsigned int        precision   = 6,
                       const std::string &       norm_suffix = "L2",
                       const unsigned int        extra_size  = 0);

  void
  print_formatted_norm(const ConditionalOStream &     pcout,
                       const std::function<double()> &compute_norm,
                       const std::string &            norm_id,
                       const std::string &            operation_name,
                       const unsigned int             precision   = 6,
                       const std::string &            norm_suffix = "L2",
                       const unsigned int             extra_size  = 0);
} // namespace MeltPoolDG::Journal
