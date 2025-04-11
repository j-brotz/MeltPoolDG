#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <chrono>


namespace MeltPoolDG::Profiling
{
  template <typename number>
  class ProfilingMonitor
  {
  public:
    ProfilingMonitor(const ProfilingData<number> &data, const TimeIterator<number> &time);

    bool
    now() const;

    void
    print(const ConditionalOStream  &pcout,
          const dealii::TimerOutput &timer,
          const MPI_Comm            &mpi_communicator) const;

  private:
    const ProfilingData<number>                       &data;
    const TimeIterator<number>                        &time;
    mutable number                                     last_written_time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> real_time_start;

    number
    compute_current_time() const;
  };
} // namespace MeltPoolDG::Profiling
