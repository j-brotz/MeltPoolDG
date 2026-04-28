#include "meltpool_vapor_flow_application.hpp"

#include <deal.II/base/mpi.h>

#include "meltpool_vapor_flow_case.hpp"

namespace MeltPoolDG::MeltPoolVaporFlow
{
  using namespace dealii;

  template <int dim, typename number>
  void
  Application<dim, number>::run()
  {
    // TODO
  }

  template <int dim, typename number>
  void
  Application<dim, number>::initialize()
  {
    // TODO
  }

  template <int dim, typename number>
  void
  Application<dim, number>::output_results(const unsigned int time_step,
                                           const number       current_time,
                                           const bool         force_output)
  {
    // TODO
  }

  template class Application<1, double>;
  template class Application<2, double>;
  template class Application<3, double>;

} // namespace MeltPoolDG::MeltPoolVaporFlow

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::MeltPoolVaporFlow::CaseParameters<double>,
                           MeltPoolDG::MeltPoolVaporFlow::Case,
                           MeltPoolDG::MeltPoolVaporFlow::Application>(argc, argv, mpi_comm);
  return 0;
}
