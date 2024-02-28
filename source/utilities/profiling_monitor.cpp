#include <meltpooldg/utilities/profiling_monitor.hpp>

#include <iostream>

namespace MeltPoolDG::Profiling
{
  template <typename number>
  ProfilingMonitor<number>::ProfilingMonitor(const ProfilingData<number> &data,
                                             const TimeIterator<number>  &time)
    : data(data)
    , time(time)
    , real_time_start(std::chrono::system_clock::now())
  {
    last_written_time = compute_current_time();
  }

  template <typename number>
  bool
  ProfilingMonitor<number>::now() const
  {
    if (!data.enable)
      return false;

    const number current_time = compute_current_time();

    const bool do_output = (current_time - last_written_time) >= data.write_time_step_size;

    if (do_output)
      last_written_time = current_time;

    return do_output;
  }

  template <typename number>
  void
  ProfilingMonitor<number>::print(const ConditionalOStream &pcout,
                                  const TimerOutput        &timer,
                                  const MPI_Comm           &mpi_communicator) const
  {
    timer.print_wall_time_statistics(mpi_communicator);
    pcout << std::endl;
    IterationMonitor::print(pcout);
    pcout << std::endl;
    DoFMonitor::print(pcout);
    pcout << std::endl;
    CellMonitor::print(pcout);
  }

  template <typename number>
  number
  ProfilingMonitor<number>::compute_current_time() const
  {
    // note: we use nanoseconds to increase the precision of the real time in seconds
    return (data.time_type == TimeType::real ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::system_clock::now() - real_time_start)
                                                   .count() /
                                                 1e9 :
                                               time.get_current_time());
  }

  template class ProfilingMonitor<double>;
} // namespace MeltPoolDG::Profiling