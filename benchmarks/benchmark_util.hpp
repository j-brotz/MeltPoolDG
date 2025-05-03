/**
 * @brief Helper to allow for running benchmarks with MPI.
 */

#pragma once

#include <deal.II/base/mpi.h>

#include <benchmark/benchmark.h>

/**
 * Benchmark reporter which does nothing. This can be used to disable specific benchmark outputs.
 * (taken from MacroAM)
 */
class NullReporter final : public benchmark::BenchmarkReporter
{
public:
  bool
  ReportContext(const Context &) override
  {
    return true;
  }

  void
  ReportRuns(const std::vector<Run> &) override
  {}

  void
  Finalize() override
  {}
};


#define MPDG_BENCHMARK_MPI_MAIN                                             \
  int main(int argc, char **argv)                                           \
  {                                                                         \
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);            \
    benchmark::Initialize(&argc, argv);                                     \
    /* TODO: How to consider load imbalance? E.g. introduce max runtime? */ \
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)      \
      benchmark::RunSpecifiedBenchmarks();                                  \
    else                                                                    \
      {                                                                     \
        NullReporter null_reporter;                                         \
        benchmark::RunSpecifiedBenchmarks(&null_reporter);                  \
      }                                                                     \
    benchmark::Shutdown();                                                  \
    return 0;                                                               \
  }
