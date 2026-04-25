#include <meltpooldg/utilities/profiling_monitor.hpp>
//
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <iostream>


namespace MeltPoolDG::Profiling
{
  template <typename number>
  ProfilingMonitor<number>::ProfilingMonitor(const ProfilingData<number>                 &data,
                                             const TimeIntegration::TimeIterator<number> &time)
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
    if (not data.enable)
      return false;

    const number current_time = compute_current_time();

    const bool do_output = (current_time - last_written_time) >= data.write_time_step_size;

    if (do_output)
      last_written_time = current_time;

    return do_output;
  }

  template <typename number>
  void
  ProfilingMonitor<number>::print(const ConditionalOStream  &pcout,
                                  const dealii::TimerOutput &timer,
                                  const MPI_Comm            &mpi_communicator) const
  {
    if (data.enable_wall_time_statistics)
      {
        timer.print_summary();
        pcout << std::endl;

        timer.print_wall_time_statistics(mpi_communicator);
        pcout << std::endl;
      }

    if (data.enable_iteration_statistics)
      {
        Journal::print_decoration_line(pcout);
        Journal::print_line(pcout, "Iteration statistics", "iteration_monitor");
        IterationMonitor<number>::print(pcout);
        pcout << std::endl;
      }

    if (data.enable_dof_statistics)
      {
        Journal::print_decoration_line(pcout);
        Journal::print_line(pcout, "DoF statistics", "dof_monitor");
        DoFMonitor<number>::print(pcout);
        pcout << std::endl;
      }

    if (data.enable_cell_statistics)
      {
        Journal::print_decoration_line(pcout);
        Journal::print_line(pcout, "Cell statistics", "cell_monitor");
        CellMonitor<number>::print(pcout);
      }
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