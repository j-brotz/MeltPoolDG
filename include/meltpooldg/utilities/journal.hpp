#pragma once
#include <meltpooldg/utilities/conditional_ostream.hpp>

#include <functional>
#include <iomanip>

namespace MeltPoolDG::Journal
{
  // maximum character width of terminal text
  static constexpr int max_text_width = 100;

  /**
   * Print a decoration line (+---+) with length limited to the maximum
   * allowed text width (max_text_width) to @p pcout.
   *
   * @param[in] pcout ConditionalOStream to print the text
   */
  inline void
  print_decoration_line(const ConditionalOStream &pcout)
  {
    pcout << "+" << std::string(max_text_width - 2, '-') << "+" << std::endl;
  }

  /**
   * Print a decorated line (starts and ends with "|") to @p pcout limited to
   * the maximum allowed text width (max_text_width) Optionally,
   * this line may contain a @p text and an identifier (@p operation_name). For
   * special characters, which in some cases do not count to the text width and
   * therefore produce ugly output, the end symbol of the decorated line may be
   * shifted via @p extra_size.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   * @param[in,opt] text Text to be printed.
   * @param[in,opt] operation_name Identifier (right aligned) of the text.
   * @param[in,opt] extra_size Shift to avoid ugly output for special characters.
   */
  void
  print_line(const ConditionalOStream &pcout,
             const std::string        &text           = "",
             const std::string        &operation_name = "",
             const unsigned int        extra_size     = 0);

  /**
   * Shorthand to print end of the simulation.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   */
  inline void
  print_end(const ConditionalOStream &pcout)
  {
    print_decoration_line(pcout);
    print_line(pcout, " end of simulation");
    print_decoration_line(pcout);
  }

  /**
   * Shorthand to print start of the simulation.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   */
  inline void
  print_start(const ConditionalOStream &pcout)
  {
    print_decoration_line(pcout);
    print_line(pcout, std::string(48, ' ') + "MeltPoolDG");
    print_decoration_line(pcout);
  }

  /**
   * Print a formatted output of a norm, given by a numerical value.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   * @param[in] norm_value Numerical value of the norm.
   * @param[in] norm_id Name of the norm, put into ||norm_id||.
   * @param[in] operation_name Identifier (right aligned) of the text.
   * @param[in,opt] precision Precision of the numerical value.
   * @param[in,opt] norm_suffix Suffix of the norm, put after ||norm_id||.
   * @param[in,opt] extra_size Shift to avoid ugly output for special characters.
   */
  void
  print_formatted_norm(const ConditionalOStream &pcout,
                       const double              norm_value,
                       const std::string        &norm_id,
                       const std::string        &operation_name,
                       const unsigned int        precision   = 6,
                       const std::string        &norm_suffix = "L2",
                       const unsigned int        extra_size  = 0);

  /**
   * Print a formatted output of a norm, given by a lambda function which
   * will be executed to compute a norm. The function is only called if
   * @p pcout is configured to produce output.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   * @param[in] norm_value Numerical value of the norm.
   * @param[in] norm_id Name of the norm, put into ||norm_id||.
   * @param[in] operation_name Identifier (right aligned) of the text.
   * @param[in,opt] precision Precision of the numerical value.
   * @param[in,opt] norm_suffix Suffix of the norm, put after ||norm_id||.
   * @param[in,opt] extra_size Shift to avoid ugly output for special characters.
   */
  void
  print_formatted_norm(const ConditionalOStream      &pcout,
                       const std::function<double()> &compute_norm,
                       const std::string             &norm_id,
                       const std::string             &operation_name,
                       const unsigned int             precision   = 6,
                       const std::string             &norm_suffix = "L2",
                       const unsigned int             extra_size  = 0);
} // namespace MeltPoolDG::Journal
