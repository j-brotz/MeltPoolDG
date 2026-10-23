#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>

#include <boost/signals2/signal.hpp>

#include <chrono>
#include <functional>
#include <string>


namespace MeltPoolDG::Profiling
{
  template <typename number>
  class ProfilingMonitor
  {
  public:
    ProfilingMonitor(const ProfilingData<number>                 &data,
                     const TimeIntegration::TimeIterator<number> &time);

    /**
     * Returns true if according to the data set in the profiling data struct, profiling outout
     * should be performed. Otherwise, returns false.
     */
    bool
    now() const;

    /**
     * Print profiling summary as obtained by the TimerOutput class and statistics summary of all
     * available monitors.
     *
     * @param pcout ConditionalOStream to print the text.
     * @param timer TimerOutput object which contains the profiling information.
     * @param mpi_communicator MPI communicator relevant for the timer.
     */
    void
    print(const ConditionalOStream  &pcout,
          const dealii::TimerOutput &timer,
          const MPI_Comm            &mpi_communicator) const;

    /**
     * Returns the verbosity of the profiling as set in the profiling data struct.
     */
    ProfilingVerbosity
    get_verbosity() const;

    /**
     * Subscribe to the profiling monitor. The subscribed callback function will be called whenever
     * the statistics summary is printed. The idea is to allow other parts of the code to add their
     * own section to the statistics summary.
     *
     * @param title Title of the section. It is prefixed by the number of the section within the
     * statistics summary and passed to @p callback.
     * @param callback The callback function printing the section to the given output stream with the
     * given title.
     *
     * @return The connection to the profiling monitor. Store it in a
     * boost::signals2::scoped_connection to unsubscribe automatically when the subscriber is
     * destroyed.
     */
    boost::signals2::connection
    subscribe(const std::string                                                          &title,
              const std::function<void(const ConditionalOStream &, const std::string &)> &callback);

  private:
    /// The profiling data struct which contains the relevant information for profiling.
    const ProfilingData<number> &data;

    /// The time iterator which is used to compute the current time.
    const TimeIntegration::TimeIterator<number> &time;

    /// The time at which the last profiling output was written.
    mutable number last_written_time = 0.0;

    /// Real time at object construction
    std::chrono::time_point<std::chrono::system_clock> real_time_start;

    /// Signal to notify subscribers when the statistics summary is printed. The section number is
    /// incremented by each subscriber.
    boost::signals2::signal<void(const ConditionalOStream &pcout, int &section_number)>
      notify_signal;

    number
    compute_current_time() const;
  };
} // namespace MeltPoolDG::Profiling
