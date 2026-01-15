#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>

#include <format>

namespace MeltPoolDG::TestUtils
{
  void
  check_for_correct_mpi_process_count(const MPI_Comm     mpi_communicator,
                                      const unsigned int expected_process_count)
  {
    const unsigned int actual_process_count =
      dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
    ASSERT_EQ(expected_process_count, actual_process_count)
      << std::format("This test is designed to run with {} MPI processes.", expected_process_count);
  }
} // namespace MeltPoolDG::TestUtils