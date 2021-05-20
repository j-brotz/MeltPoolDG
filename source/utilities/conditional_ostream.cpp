#include <meltpooldg/utilities/conditional_ostream.hpp>

namespace MeltPoolDG
{
  ConditionalOStream::ConditionalOStream(std::ostream &stream, const bool active)
    : dealii::ConditionalOStream(stream, active)
    , output_stream(stream)
    , f(stream.flags())
  {}

  ConditionalOStream::~ConditionalOStream()
  {
    output_stream.flags(f);
  }
} // namespace MeltPoolDG
