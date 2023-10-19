#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/journal.hpp>


namespace MeltPoolDG::Journal
{
  static const std::string start_line_symbol = "| "; // move to cpp namespace
  static const std::string end_line_symbol   = " |";

  void
  print_line(const ConditionalOStream &pcout,
             const std::string        &text,
             const std::string        &operation_name,
             const unsigned int        extra_size)
  {
    std::ostringstream str;
    str << operation_name << end_line_symbol;
    pcout << start_line_symbol << text << std::right
          << std::setw(max_text_width - text.length() - end_line_symbol.length() + extra_size)
          << str.str() << std::endl;
  }

  void
  print_formatted_norm(const ConditionalOStream &pcout,
                       const double              norm_value,
                       const std::string        &norm_id,
                       const std::string        &operation_name,
                       const unsigned int        precision,
                       const std::string        &norm_suffix,
                       const unsigned int        extra_size)
  {
    std::ostringstream str;
    str << "|| " << norm_id << " ||" << norm_suffix << " = " << std::setprecision(precision)
        << std::left << std::scientific << norm_value;

    print_line(pcout, str.str(), operation_name, extra_size);
  }

  void
  print_formatted_norm(const ConditionalOStream      &pcout,
                       const std::function<double()> &compute_norm,
                       const std::string             &norm_id,
                       const std::string             &operation_name,
                       const unsigned int             precision,
                       const std::string             &norm_suffix,
                       const unsigned int             extra_size)
  {
    if (pcout.now() == false)
      return;
    else
      {
        print_formatted_norm(
          pcout, compute_norm(), norm_id, operation_name, precision, norm_suffix, extra_size);
      }
  }
} // namespace MeltPoolDG::Journal
