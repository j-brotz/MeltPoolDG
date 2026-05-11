#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>
#include <cfenv>


int
main(int argc, char **argv)
{
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  ::testing::InitGoogleTest(&argc, argv);
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  return RUN_ALL_TESTS();
}
