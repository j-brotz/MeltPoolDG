#include "benchmark_util.hpp"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <ranges>
#include <string>

// from google benchmark src
#include "colorprint.h"
#include "string_util.h"

MeltPoolDG::MPIConsoleReporter::MPIConsoleReporter(const MPI_Comm communicator)
  : communicator(communicator)
{}

bool
MeltPoolDG::MPIConsoleReporter::ReportContext(const Context &context)
{
  bool proceed;
  if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
    proceed = ConsoleReporter::ReportContext(context);
  MPI_Bcast(&proceed, 1, MPI_CXX_BOOL, 0, communicator);
  return proceed;
}

void
MeltPoolDG::MPIConsoleReporter::PrintHeader(const Run &run)
{
  if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
    {
      std::string str =
        benchmark::FormatString("%-*s", static_cast<int>(name_field_width_), "Benchmark");
      for (unsigned int i = 0; i < dealii::Utilities::MPI::n_mpi_processes(communicator); ++i)
        {
          std::string time_str = "Time Rank " + std::to_string(i);
          str += benchmark::FormatString(" %16s", time_str.c_str());
        }

      const std::string iter = "Iterations";
      str += benchmark::FormatString(" %14s", iter.c_str());
      if (!run.counters.empty())
        {
          if ((output_options_ & OO_Tabular) != 0)
            {
              for (auto const &name : std::views::keys(run.counters))
                {
                  str += benchmark::FormatString(" %10s", name.c_str());
                }
            }
          else
            {
              str += " UserCounters...";
            }
        }

      const auto line = std::string(str.length(), '-');
      GetOutputStream() << line << "\n" << str << "\n" << line << "\n";
    }
}

void
MeltPoolDG::MPIConsoleReporter::PrintRunData(const Run &result)
{
  // Helper function taken from the Google Benchmark library (console_reporter.cc). As it is
  // declared as static in the library, we cannot use the funciton here but have to define our own.
  auto format_time = [](const double time) -> std::string {
    // For the time columns of the console printer, 16 digits are reserved. One of
    // them is a space and max two of them are the time unit (e.g. ns). That puts
    // us at 13 digits usable for the number.

    // Align decimal places...
    if (time < 1.0)
      {
        return benchmark::FormatString("%13.3f", time);
      }
    if (time < 10.0)
      {
        return benchmark::FormatString("%13.2f", time);
      }
    if (time < 100.0)
      {
        return benchmark::FormatString("%13.1f", time);
      }
    // Assuming the time is at max 9.9999e+99, and we have 13 digits for the
    // number, we get 10-1(.)-1(e)-1(sign)-2(exponent) = 5 digits to print.
    if (time > 9999999999999 /*max 13-digit number*/)
      {
        return benchmark::FormatString("%1.4e", time);
      }
    return benchmark::FormatString("%13.0f", time);
  };

  // collect the time measurements on rank 0
  const double        real_time = result.GetAdjustedRealTime();
  std::vector<double> collected_real_times(dealii::Utilities::MPI::n_mpi_processes(communicator));
  MPI_Gather(
    &real_time, 1, MPI_DOUBLE, collected_real_times.data(), 1, MPI_DOUBLE, 0, communicator);

  // print benchmark results to the console
  if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
    {
      typedef void(PrinterFn)(std::ostream &, benchmark::LogColor, const char *, ...);
      auto      &Out     = GetOutputStream();
      const auto printer = static_cast<PrinterFn *>(benchmark::ColorPrintf);

      // print the name of the benchmark
      printer(
        Out, benchmark::COLOR_GREEN, "%-*s ", name_field_width_, result.benchmark_name().c_str());

      if (result.run_type != Run::RT_Aggregate ||
          result.aggregate_unit == benchmark::StatisticUnit::kTime)
        {
          // print the real time measurements
          const char *timeLabel = GetTimeUnitString(result.time_unit);
          for (const auto local_real_time : collected_real_times)
            {
              const std::string real_time_str = format_time(local_real_time);
              printer(Out, benchmark::COLOR_YELLOW, "%s %-2s ", real_time_str.c_str(), timeLabel);
            }
        }
      else
        {
          // not supported
          AssertThrow(
            false,
            dealii::ExcMessage(
              "Specified output not supported! Currently only real time output with the number of"
              " iterations and custom user counter are supported!"));
        }

      // print the number of iterations
      printer(Out, benchmark::COLOR_CYAN, "%14lld", result.iterations);

      // print custom counters
      for (const auto &[name, counter] : result.counters)
        {
          const std::size_t cNameLen = std::max(static_cast<std::size_t>(10), name.length());
          std::string       s;
          const auto       *unit = "";
          s                      = benchmark::HumanReadableNumber(counter.value, counter.oneK);
          if ((counter.flags & benchmark::Counter::kIsRate) != 0)
            {
              unit = (counter.flags & benchmark::Counter::kInvert) != 0 ? "s" : "/s";
            }

          printer(
            Out, benchmark::COLOR_DEFAULT, " %*s%s", cNameLen - strlen(unit), s.c_str(), unit);
        }

      // print a new line
      printer(Out, benchmark::COLOR_DEFAULT, "\n");
    }
}