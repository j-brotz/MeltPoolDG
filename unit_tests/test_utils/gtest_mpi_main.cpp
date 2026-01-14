#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>


int
main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  return RUN_ALL_TESTS();
}
