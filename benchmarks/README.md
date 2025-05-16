# Using the Benchmarking Framework

The benchmarks/ directory contains everything needed to set up and run custom benchmarks to evaluate and improve
specific parts of the MeltPoolDG library. This framework is based on Google Benchmark, and standard usage is consistent
with the official Google Benchmark user guide.

## Enabling Benchmarking

To use the benchmarking framework, compile MeltPoolDG with the MPDG_ENABLE_BENCHMARKING flag enabled. You don’t need to
manually install any additional dependencies — CMake will handle everything for you. Setting this flag instructs CMake
to:

- Install the required benchmark dependencies
- Generate build targets for all benchmarks found in subfolders of benchmarks/

## Adding New Benchmarks

If you create a new subfolder inside benchmarks/, make sure it includes a CMakeLists.txt that calls the function:

```
mpdg_auto_setup_benchmarks()
```

This function automatically generates an executable for each .cpp file in that folder. In most cases, this is all you
need. If you have more specific setup requirements, you can customize the CMakeLists.txt as needed.

For general instructions on writing benchmark executables, refer to the Google Benchmark User Guide.

## Benchmarking MPI Code

### Using a Single MPI Process

When benchmarking MPI-based code with one process, only one change is required compared to standard Google Benchmark
usage:

- Include `benchmark_util.h`
- Replace `BENCHMARK_MAIN()` with `MPDG_BENCHMARK_MPI_MAIN`

This ensures that the MPI environment is properly initialized before any MPI-related functionality is used.

### Using Multiple MPI Processes

Benchmarking code with multiple MPI processes is a bit more involved—and more restricted — than single-process
benchmarking. However, from the user perspective, only minimal changes are needed in the benchmark executable.

Instead of the typical Google Benchmark loop:

```cpp
for (auto _ : state)
    // code to be benchmarked
```

you must use a helper function provided in `benchmark_util.hpp`:

```cpp
void mpi_benchmark_loop(benchmark::State &state, const std::function<void()> &benchmark_function)
```

where `state` is the `Benchmark::State` object used by the current benchmark and `benchmark_function` is the code
you want to benchmark wrapped in a `std::function<void()>`.

#### Why This Change Is Necessary

In the default Google Benchmark setup, each process independently determines how many iterations to run based on its
local timing. In an MPI environment, this can lead to problems—some processes may exit the loop earlier than others,
potentially causing deadlocks if remaining processes are still running collective operations.

The mpi_benchmark_loop() function avoids this by synchronizing iteration counts across all processes using an
MPI_Allreduce. It ensures that all ranks run the same number of iterations, making the benchmark results statistically
stable and deadlock-free.

#### Current Limitations of Multiple Process MPI Benchmarks

While the multi-process MPI benchmarking system in MeltPoolDG works reliably, there are a few limitations you should be
aware of:

- Reduced Output: Compared to single-process benchmarks, the console output is minimal.
    - Only the wall-clock time, number of iterations, and benchmark name are printed.
    - Any custom counters are also printed — but only those from rank 0.

However, the key advantage is that the benchmark reports individual wall-clock times for each MPI process. This provides
valuable insights into effects like load imbalance.
