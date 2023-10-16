/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, TUM, March 2021
 *
 * ---------------------------------------------------------------------*/

#pragma once
#include <deal.II/base/conditional_ostream.h>

namespace MeltPoolDG
{
  using namespace dealii;

  class ConditionalOStream : public dealii::ConditionalOStream
  {
  public:
    ConditionalOStream(const dealii::ConditionalOStream &conditional_o_stream);

    ConditionalOStream(const ConditionalOStream &conditional_o_stream);

    ConditionalOStream(std::ostream &stream, const bool active = true);

    ~ConditionalOStream();

    bool
    now() const;

  private:
    std::ostream           &output_stream;
    std::ios_base::fmtflags f;
    const bool              any_is_active = false;
  };

} // namespace MeltPoolDG
