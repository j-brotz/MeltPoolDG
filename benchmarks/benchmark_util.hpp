/**
 * @brief Helper to allow for running benchmarks with MPI.
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include "benchmark/benchmark.h"

namespace MeltPoolDG
{

  /**
   * @brief Executes a Google Benchmark loop across multiple MPI processes.
   *
   * In standard usage, Google Benchmark determines the number of iterations based on local timing
   * on each process. In an MPI context, this can result in different processes exiting the
   * benchmark loop after different number of iterations, potentially causing deadlocks if some are
   * still running while others have exited.
   *
   * This function ensures synchronized iteration counts across all processes by using an
   * MPI_Allreduce to compute a common maximum iteration count. To use the funciton, the user
   * provides the benchmark state and a function to be benchmarked, which will be executed once per
   * iteration.
   *
   * @param state               The Google Benchmark state object.
   * @param benchmark_function  The function to benchmark, executed in each iteration.
   *
   * @note In cases of load imbalance, benchmarking may take longer since the process with the
   * fastest iteration time (and thus requiring the most iterations) will determine the total number
   * of iterations for all processes. While this may increase runtime, it is necessary to ensure
   * statistically stable and consistent results across all MPI processes.
   */
  inline void
  mpi_benchmark_loop(benchmark::State &state, const std::function<void()> &benchmark_function)
  {
    auto *max_iterations = const_cast<int64_t *>(&state.max_iterations);
    MPI_Allreduce(max_iterations, max_iterations, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
    for (auto _ : state)
      {
        benchmark_function();

        // If the benchmarked code involves MPI synchronization, faster processes may wait for
        // slower ones still in previous iterations, distorting timing results. Inserting a barrier
        // here ensures all processes begin the next iteration in sync.
        state.PauseTiming();
        MPI_Barrier(MPI_COMM_WORLD);
        state.ResumeTiming();
      }
  }

  /**
   * @brief Benchmark reporter class for console output in multiprocess MPI benchmarking.
   *
   * This class, built on Google Benchmark's ConsoleReporter, supports MPI-based benchmarks. It
   * prints individual timing results from each MPI process to the console.
   */
  class MPIConsoleReporter final : public benchmark::ConsoleReporter
  {
  public:
    /**
     * Constructor that stores the provided MPI communicator.
     *
     * @param communicator MPI communicator used during the benchmark.
     */
    explicit MPIConsoleReporter(const MPI_Comm communicator);

    /**
     * Print context information to the console (e.g., cache sizes). This function still calls the
     * corresponding ReportContext function from the ConsoleReporter base class but ensures that
     * only MPI rank 0 writes to the console.
     *
     * @param context Struct providing access to the context information.
     * @return Must return true if the benchmark should start. Otherwise, the Google Benchmark
     * library will skip execution.
     */

    bool
    ReportContext(const Context &context) override;

  protected:
    /**
     * @brief Print a structured header based on the provided benchmark run information.
     *
     * @param run Struct containing information about the benchmark run.
     */
    void
    PrintHeader(const Run &run) override;

    /**
     * @brief Print formatted benchmark results for a single run.
     *
     * Outputs timing and metric data from the given Run object, aligned with the header previously
     * printed by PrintHeader.
     *
     * @param result Struct containing the results of a benchmark run.
     */
    void
    PrintRunData(const Run &result) override;

  private:
    // MPI Communicator used for running the benchmarks.
    const MPI_Comm communicator;
  };

#define MPDG_BENCHMARK_MPI_MAIN                                           \
  int main(int argc, char **argv)                                         \
  {                                                                       \
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);          \
    benchmark::Initialize(&argc, argv);                                   \
    if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)     \
      {                                                                   \
        benchmark::RunSpecifiedBenchmarks();                              \
      }                                                                   \
    else if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1) \
      {                                                                   \
        MeltPoolDG::MPIConsoleReporter reporter(MPI_COMM_WORLD);          \
        benchmark::RunSpecifiedBenchmarks(&reporter);                     \
      }                                                                   \
    else                                                                  \
      AssertThrow(false, dealii::ExcInternalError());                     \
    benchmark::Shutdown();                                                \
    return 0;                                                             \
  }
} // namespace MeltPoolDG
