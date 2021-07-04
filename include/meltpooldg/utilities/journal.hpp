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

  template <int dim>
  void
  print_mesh_information(const ConditionalOStream &pcout,
                         const ScratchData<dim> &  scratch_data,
                         const unsigned int        dof_idx        = 0,
                         const std::string &       operation_name = "")
  {
    std::ostringstream str;
    str << "#dofs: " << scratch_data.get_dof_handler(dof_idx).n_dofs();
    auto vec =
      Utilities::MPI::gather(MPI_COMM_WORLD, scratch_data.get_triangulation().n_active_cells());

    int sum_cells = 0;
    for (auto &i : vec)
      sum_cells += i;

    str << "     #cells: " << sum_cells;

    print_line(pcout, str.str(), operation_name);
  }

} // namespace MeltPoolDG::Journal
