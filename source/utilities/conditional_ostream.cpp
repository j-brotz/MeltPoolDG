#include <deal.II/base/mpi.h>

#include <meltpooldg/utilities/conditional_ostream.hpp>

namespace MeltPoolDG
{
  ConditionalOStream::ConditionalOStream(const dealii::ConditionalOStream &other)
    : ConditionalOStream(other.get_stream(), other.is_active())
  {}

  ConditionalOStream::ConditionalOStream(std::ostream &stream, const bool active)
    : dealii::ConditionalOStream(stream, active)
    , output_stream(stream)
    , f(stream.flags())
    , any_is_active(
        static_cast<bool>(Utilities::MPI::max(static_cast<int>(active), MPI_COMM_WORLD)))
  {}

  ConditionalOStream::~ConditionalOStream()
  {
    output_stream.flags(f);
  }

  bool
  ConditionalOStream::now() const
  {
    return any_is_active;
  }
} // namespace MeltPoolDG
